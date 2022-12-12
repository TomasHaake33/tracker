#include "Header.h"

Track::Track(const cv::Rect& coords,
	const size_t id,
	const Box& bgBox,
	const size_t trackerId,
	const cv::Mat hist) :
	m_coords(coords), m_id(id), m_trackerId(trackerId), m_hist(hist)
{
	m_lastPositions.push_back({ bgBox.m_coords.x, bgBox.m_coords.y });
}

Box MyTracker::searchBox()
{

	m_bgSub->apply(m_frame, m_fgMask);

	cv::erode(m_fgMask, m_fgMask, cv::Mat(), cv::Point(-1, -1), 1);
	cv::dilate(m_fgMask, m_fgMask, cv::Mat(), cv::Point(-1, -1), 2);


	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(m_fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<Box> boxes;

	for (auto& ctr : contours)
	{
		if (cv::contourArea(ctr) > SEARCH_BOX_AREA) 
		{
			Box box(cv::boundingRect(ctr));
			boxes.emplace_back(box);
		}
	}

	Box largest;

	for (int i = 0; i < boxes.size(); i++)
	{
		if (boxes[i].m_coords.area() >= largest.m_coords.area())
			largest = boxes[i];
	}

	return largest;
}

bool MyTracker::isStill() const
{

	if (m_track->m_lastPositions.size() < STILL_FRAMES)
		return false;

	/*Считаем количество точек, которые находятся в определенном
	радиусе от самой первой. Если таких точек STILL_FRAMES, вернем
	true*/
	else 
	{
		auto pointsInRadius = 0;

		auto& first_point = m_track->m_lastPositions.front();
		for (auto& point : m_track->m_lastPositions)
		{
			auto dist = sqrt(pow(point.front() - first_point.front(), 2) + 
				(pow(point.back() - first_point.back(), 2)));
			if (dist < STILL_RADIUS)
				pointsInRadius += 1;
		}

		if (pointsInRadius == STILL_FRAMES)
			return true;
		else
			return false;
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
	{
		return 0;
	}
}

void MyTracker::updateTrack()
{

	if (m_trackList.size() > 0)
	{
		for (auto& track : m_trackList) 
		{

			if (!track->m_isPresent)
				track->m_liveFrames -= 1;

			if (track->m_liveFrames <= 0) 
			{
				track->m_liveFrames = 0;
				track->m_expired = true;
			}
		}
	}
}

void MyTracker::updateBox()
{

	Box tempBox = searchBox();
	if (m_track != nullptr) 
	{
		if (IOU(tempBox.m_coords, m_track->m_coords) > IOU_THRESHOLD)
		{
			m_bgBox.m_coords = cv::Rect();
			m_bgBox.m_framesAlive = 0;
			return;
		}

		else 
		{
			if (tempBox.m_coords != cv::Rect() && 
				IOU(tempBox.m_coords, m_bgBox.m_coords) > IOU_THRESHOLD)
			{
				m_bgBox.m_coords = tempBox.m_coords;
				m_bgBox.m_framesAlive += 1;
				return;
			}
			else 
			{
				m_bgBox.m_coords = cv::Rect();
				m_bgBox.m_framesAlive = 0;
				return;
			}
			return;
		}
	}

	else 
	{
		m_bgBox.m_coords = tempBox.m_coords;
		if (IOU(tempBox.m_coords, m_bgBox.m_coords) > IOU_THRESHOLD)
			m_bgBox.m_framesAlive += 1;
		else
			m_bgBox.m_framesAlive = 0;
		return;
	}

}

void MyTracker::initTracker()
{
	/*Трекер не инициализируем, если бокс еще не живет достаточно долго*/
	if (m_bgBox.m_framesAlive < ACTIVATION_FRAMES)
		return;

	/*Если дошли до сюда, значит надо занулить активный трек.
	То есть, временно считаем, что теркер не следит ни за кем*/
	if (m_track != nullptr)
	{
		m_track->m_isPresent = false;
		m_track = nullptr;
	}

	/*Среди всех не пропавших (expired) треков ищем такой, который больше остальных похож
	по цвету на текущий трек.*/
	double largest = 0;
	std::shared_ptr<Track> mostSimilar = nullptr;
	for (auto& compTrack : m_trackList) {  
		if (!compTrack->m_isPresent && !compTrack->m_expired) 
		{
			auto comp = compHist(compTrack);
			if (comp > largest)
			{
				largest = comp;
				mostSimilar = compTrack;
			}
		}
	}

	/*Если совпадение больше порога, присваиваем текущему треку совпавший. Также, обновляем
	остальные члены структуры*/
	if (largest > HIST_THRESHOLD)
	{
		m_bgBox.m_framesAlive = 0;
		m_track = mostSimilar;
		m_track->m_coords = m_bgBox.m_coords;
		m_track->m_hist = calcBoxHist();
		m_track->m_lastPositions.clear();
		m_track->m_lastPositions.push_back({ m_bgBox.m_coords.x, m_bgBox.m_coords.y });
		m_track->m_isPresent = true;
		m_track->m_liveFrames = LIVE_FRAMES;
		m_track->m_trackerId = m_trackerId;
		m_pointer = cv::TrackerKCF::create();
		m_pointer->init(m_frame, m_track->m_coords);
		return;
	}

	/*Если сопадение по цвету не нашли, создаем новый трек и добавляем его в 
	trackList. Трекер будет следить за этим треком*/
	m_track.reset(new Track(m_bgBox.m_coords, m_trackList.size(), m_bgBox, m_trackerId,
		calcBoxHist()));
	m_trackList.emplace_back(m_track);
	m_pointer = cv::TrackerKCF::create();
	m_pointer->init(m_frame, m_track->m_coords);
	return;
}

void MyTracker::drawTracks()
{
	for (auto& track : m_trackList)
	{
		short id = track->m_trackerId;
		if (!track->m_expired && id == m_trackerId && track->m_liveFrames > LIVE_FRAMES - 30)
		{
			cv::rectangle(m_frame, cv::Point(track->m_coords.x, track->m_coords.y),
				cv::Point(track->m_coords.x + track->m_coords.width, track->m_coords.y + track->m_coords.height),
				cv::Scalar(0, 255, 0));
			std::string text = "ID: ";
			text += std::to_string(track->m_id);
			cv::putText(m_frame, text, cv::Point(track->m_coords.x, track->m_coords.y + track->m_coords.height / 2), 
				cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
			/*cv::putText(m_frame, std::to_string(track->m_liveFrames),
				cv::Point(track->m_coords.x, track->m_coords.y + track->m_coords.height / 3),
				cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));*/
		}
	}
}

cv::Mat MyTracker::calcBoxHist() const
{

	std::vector<std::vector<cv::Point>> boxPoints = {
		{ cv::Point(m_bgBox.m_coords.x, m_bgBox.m_coords.y),
		cv::Point(m_bgBox.m_coords.x + m_bgBox.m_coords.width, m_bgBox.m_coords.y),
		cv::Point(m_bgBox.m_coords.x, m_bgBox.m_coords.y + m_bgBox.m_coords.height),
		cv::Point(m_bgBox.m_coords.x + m_bgBox.m_coords.width, m_bgBox.m_coords.y + m_bgBox.m_coords.height)} };

	cv::Mat mask(m_frame.rows, m_frame.cols, CV_8UC1, cv::Scalar(0));
	cv::fillPoly(mask, boxPoints, cv::Scalar(255));

	cv::Mat splitted[3];
	cv::split(m_frame, splitted);

	int histSize[] = { 256, 256, 256 };
	int channels[] = { 0, 1, 2 };
	float branges[] = { 0, 255 };
	float granges[] = { 0, 255 };
	float rranges[] = { 0, 255 };
	const float* histRanges[] = {branges, granges, rranges};

	cv::Mat histogram;
	cv::calcHist(splitted, 3, channels, mask, histogram, 3, histSize, histRanges);
	cv::normalize(histogram, histogram, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

	return histogram;
}

double MyTracker::compHist(const std::shared_ptr<Track> compTrack) const
{

	auto boxHist = calcBoxHist();
	double comp = std::abs(cv::compareHist(boxHist, compTrack->m_hist, cv::HISTCMP_CORREL));
	std::cout << comp << std::endl;

	if (comp >= HIST_THRESHOLD)
		return true;
	else
		return false;

}
