#include "Header.h"

void Logger::log(Severity severity, const char* msg) noexcept {
	if (severity <= Severity::kWARNING) {
		std::cout << msg << std::endl;
	}
}

bool NNet::buildEngine(const char modelPath[]) {

	std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(m_logger));

	uint32_t flag = 1U << static_cast<uint32_t>
		(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

	std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, m_logger));
	parser->parseFromFile(modelPath,
		static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));

	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		std::cout << parser->getError(i)->desc() << std::endl;


	const auto input = network->getInput(0);
	const auto output = network->getOutput(0);
	const auto inputName = input->getName();
	const auto inputDims = input->getDimensions();
	int32_t inputC = inputDims.d[1];
	int32_t inputH = inputDims.d[2];
	int32_t inputW = inputDims.d[3];

	std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
	profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, inputC, inputH, inputW));
	profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(1, inputC, inputH, inputW));
	config->addOptimizationProfile(profile);

	
	std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
	if (!(serializedModel)) return false;

	std::ofstream ofs("../Model.engine", std::ios::binary);
	ofs.write((char*)(serializedModel->data()), serializedModel->size());
	ofs.close();

	return true;
}

bool NNet::load() {
	
	std::ifstream ifs(ENGINE_PATH, std::ios::binary | std::ios::ate);
	size_t engine_size = ifs.tellg();
	ifs.seekg(0, ifs.beg);

	std::vector<char> modelStream;
	modelStream.resize(engine_size);
	ifs.read(modelStream.data(), engine_size);

	std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(m_logger));
	if (!(runtime)) return false;

	m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelStream.data(), engine_size));
	if (!(m_engine)) return false;

	m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
	if (!(m_context)) return false;
	
	return true;
}

bool NNet::infer(const cv::Mat& img, std::vector<float>& features) {

	//Смотрю, какие сеть ожидает входы и делаю ресайз буфера
	auto dims = m_engine->getBindingDimensions(0);
	auto outputL = (m_engine->getBindingDimensions(1).d[1]) * (m_engine->getBindingDimensions(1).d[0]);
	Dims4 inputDims = { int32_t(1), dims.d[1], dims.d[2], dims.d[3] };

	m_inputBuff.hostBuffer.resize(inputDims);
	m_inputBuff.deviceBuffer.resize(inputDims);

	//Аналогично с выходами
	Dims3 outputDims{ int32_t(1), m_engine->getBindingDimensions(1).d[1], m_engine->getBindingDimensions(1).d[0] };
	m_outputBuff.hostBuffer.resize(outputDims);
	m_outputBuff.deviceBuffer.resize(outputDims);

	//Копирую входной кадр на gpu
	cudaMemcpy(m_inputBuff.deviceBuffer.data(), img.ptr<float>(0), m_inputBuff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
	
	std::vector<void*> predicitonBindings = { m_inputBuff.deviceBuffer.data(), m_outputBuff.deviceBuffer.data() };

	bool status = m_context->executeV2(predicitonBindings.data());
	if (!status) return false;

	//Возвращаю обратно результаты инференса
	cudaMemcpy(m_outputBuff.hostBuffer.data(), m_outputBuff.deviceBuffer.data(), 
		m_outputBuff.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);

	features.resize(outputL);

	//Записываю результаты в вектор
	memcpy(features.data(), reinterpret_cast<const char*>(m_outputBuff.hostBuffer.data()), outputL * sizeof(float));
	
	return true;
}

void MyTracker::nms(double thresh, int neighbors)
{

	m_outRects.clear();

	const size_t size = m_scores.size();
	std::multimap<float, size_t> sorted;
	for (size_t i = 0; i < size; ++i)
		sorted.emplace(m_scores[i], i);

	while (sorted.size() > 0) {

		auto highest = --std::end(sorted);
		const cv::Rect& rect1 = m_rects[highest->second];

		int neighborsCount = 0;

		sorted.erase(highest);

		for (auto iter = std::begin(sorted); iter != std::end(sorted);) {

			const cv::Rect& rect2 = m_rects[iter->second];

			if (IOU(rect1, rect2) > thresh) {
				iter = sorted.erase(iter);
				++neighborsCount;
			}
			else {
				++iter;
			}

		}

		if (neighborsCount >= neighbors)
			m_outRects.push_back(rect1);

	}

}

void MyTracker::processOutputs(const double thresh, const cv::Mat& frame) {

	for (int i = 0; i < 8732; ++i) {
		float score = m_rawOutputs[i * 6 + 5];
		if (score > thresh) {
			cv::Point p1(m_rawOutputs[i * 6] * frame.cols, m_rawOutputs[i * 6 + 1] * frame.rows);
			cv::Point p2(m_rawOutputs[i * 6 + 2] * frame.cols, m_rawOutputs[i * 6 + 3] * frame.rows);
			m_rects.push_back(cv::Rect(p1, p2));
			m_scores.push_back(score);
		}
	}

}

double MyTracker::IOU(const cv::Rect& rect1, const cv::Rect& rect2) const
{

	if (rect1.area() != 0 && rect2.area() != 0) 
	{
		double intArea = (rect1 & rect2).area();
		double totalArea = rect1.area() + rect2.area() - intArea;
		return (intArea / totalArea) * 100;
	}
	else
		return 0;
}

void MyTracker::updateTracks()
{

	if (m_tracks.size() > 0) {
		
		/*Определяем наибольший id среди всех треков*/
		int largestId = m_tracks[0]->m_id;
		for (auto& track : m_tracks)
		{
			if (track->m_id > largestId)
				largestId = track->m_id;
		}

		/*Для каждого непропавшего (expired) трека, ищу по всем
		выходам сети пересекающиеся с треком. Обновляю трек в зависимости
		от результата*/
		for (auto& track : m_tracks) 
		{ 
			if (track->m_expired)
				continue;
			
			/*Флаг нужен, чтобы понять, выполнился ли один из if-ов в цикле
			for. Такое возможно, только если выход сети пересекся с треком*/
			bool trackFlag = false;
			if (m_outRects.size() > 0) {
				for (auto& output : m_outRects) {
					auto iou = IOU(output, track->m_box);
					/*Если трек активирован и пересекается с выходом сети,
					обновляю его координаты соответственно*/
					if (iou > 20 && track->m_activated) {
						track->m_liveFrames = LIVE_FRAMES;
						track->m_box = output;
						track->m_present = true;
						trackFlag = true;
						break;
					}
					/*Если трек еще не активирован, но есть пересечение с выходом
					сети, обновляю координаты. Увеличиваю время жизни (эквивалентно
					уменьшению оставшегося времени до активации). Если пора активировать трек,
					то делаю это и обновляю остальные члены структуры.*/
					else if (iou > 20 && !track->m_activated) {
						track->m_box = output;
						if (++track->m_actvFrames >= ACTIVATION_FRAMES) {
							track->m_liveFrames = LIVE_FRAMES;
							track->m_actvFrames = 0;
							track->m_activated = true;
							track->m_present = true;
							track->m_id = largestId + 1;
						}
						trackFlag = true;
						break;
					}
				}

				/*Если трек не пересекся ни с одним выходом, помечаю его пропавшим*/
				if (!trackFlag)
					track->m_present = false;
			}
			
			/*Если дошли до сюда, значит сеть не дала ни один выход. Считаю трек пропавшим*/
			else {
				if (track->m_activated) {
					track->m_present = false;
					--track->m_liveFrames;
				}
				else {
					track->m_actvFrames = 0;
				}
				
			}
		}

		/*Ищу индексы выходов в векторе outputs, которые
		не пересекаются ни с какими треками. Создаю из них треки*/
		std::vector<int> inds = searchNew(m_outRects);
		for (int i = 0; i < inds.size(); ++i)
		{
			m_tracks.emplace_back(std::make_unique<Track>(m_outRects[inds[i]]));
		}

		/*Уменьшаю время жизни исчезнувших треков. Если время вышло, помечаю
		их пропавшими*/
		for (auto& track : m_tracks) {
			if (!track->m_present && track->m_activated)
				--track->m_liveFrames;
			if (track->m_liveFrames <= 0) {
				track->m_liveFrames = 0;
				track->m_expired = true;
			}
		}
	}

	/*Если дошли до сюда, значит нет ни одного трека. Создаем новые треки из выходов*/
	else 
	{
		for (int i = 0; i < m_outRects.size(); ++i)
			m_tracks.emplace_back(std::make_unique<Track>(m_outRects[i], m_tracks.size()));
	}

}


std::vector<int> MyTracker::searchNew(const std::vector<cv::Rect>& outputs) const
{

	std::vector<int> inds;
	for (auto i = 0; i < outputs.size(); ++i) {
		bool flag = false;
		for (auto& track : m_tracks) {
			if (track->m_expired)
				continue;

			if (IOU(outputs[i], track->m_box) > 0) {
				flag = true;
				break;
			}
		}
		if (!flag)
			inds.push_back(i);
	}

	return inds;

}

bool MyTracker::drawTracks(cv::Mat& frame)
{
	if (m_tracks.size() == 0)
		return false;

	for (auto& track : m_tracks) {
		
		if (track->m_expired || !track->m_activated)
			continue;

		cv::Point p1(track->m_box.x, track->m_box.y);
		cv::Point p2(track->m_box.x + track->m_box.width,
			track->m_box.y + track->m_box.height);
		cv::rectangle(frame, p1, p2, { 0, 255, 0 });

		std::string text = "ID: ";
		text += std::to_string(track->m_id);
		cv::putText(frame, text, cv::Point(track->m_box.x, track->m_box.y + track->m_box.height / 2),
			cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
	}

	return true;
}

void MyTracker::clearOutputs()
{
	m_rawOutputs.clear();
	m_rects.clear();
	m_scores.clear();
	m_outRects.clear();
}