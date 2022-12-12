#include "Header.h"

int main()
{

	MyTracker tracker;

	std::cout << "Type Y to build an engine or N no use existing engine: ";
	char ans;
	std::cin >> ans;

	if (ans == 'Y' || ans == 'y')
		tracker.buildEngine(MODEL_PATH);
	else if (ans == 'N' || ans == 'n') { }
	else { throw; }

	tracker.loadModel();

	cv::VideoCapture video(VIDEO_PATH);
	cv::Mat frame;

	using FPS = std::chrono::duration<uint64_t, std::ratio<1, 30>>;

	auto start = std::chrono::system_clock::now();
	while (video.read(frame)) 
	{
		auto curTime = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<FPS> (curTime - start);
		/*Если не прошло UPDATE_RATE кадров (точнее времени для стольких кадров),
		просто рисую треки*/
		if (elapsed % FPS(UPDATE_RATE) >= FPS(1))
		{
			tracker.drawTracks(frame);
			cv::waitKey(1);
			cv::imshow("1", frame);

			std::this_thread::sleep_for(FPS(1));
			continue;
		}

		auto curTimeInfer = std::chrono::system_clock::now();
		/*Транформирую кадр в подходящий для нейросети формат. То есть NCHW размерность,
		размер 300x300, средние по каналам (123, 117, 104) и свап B,R каналов, т.к.
		opencv считывает BGR*/
		cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, { 300, 300 }, { 123.0, 117.0, 104.0 }, true);
		
		tracker.inferModel(blob);
		tracker.processOutputs(0.2, frame);
		tracker.nms(50, 1);

		tracker.updateTracks();

		tracker.drawTracks(frame);
		tracker.clearOutputs();
		cv::waitKey(1);
		cv::imshow("1", frame);
		auto elapsedInfer = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now() - curTime);
		/*Чтобы визуально сохранялся фпс, делаю этот шаг. Аргумент при длительном инференсе
		может стать отрицательным, для читаемости эту проверку не делаю*/
		std::this_thread::sleep_for(FPS(1) - elapsedInfer);
	}
}