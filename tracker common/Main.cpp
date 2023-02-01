#include "Header.h"

int main() {

	std::vector<cv::VideoCapture> video { cv::VideoCapture(DEFAULT_PATH2), cv::VideoCapture(DEFAULT_PATH1) };
	std::vector<cv::Mat> frame { cv::Mat(), cv::Mat() };

	/*Массив содержит указатели на все существовавшие треки, пригодится для хранения
	id треков и сравнения похожести. Не чищу его, т.к. он служит также в качестве своеборазной БД.*/
	std::vector<std::shared_ptr<Track>> trackList;
	std::vector<MyTracker> trackers{ MyTracker(0, trackList, frame[0]), MyTracker(1, trackList, frame[1])};

	/*Основная единица времени, равная (1 / FPS_видео)*/
	using FPS = std::chrono::duration<uint64_t, std::ratio<1, 30>>;

	auto start = std::chrono::system_clock::now();
	while (video[0].read(frame[0]) && video[1].read(frame[1]))
	{
		auto curTime = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<FPS> (curTime - start);

		/*Если не прошло времени для UPDATE_RATE кадров с момента start, 
		просто отображаем треки и продолжаем цикл.*/
		if (elapsed % FPS(UPDATE_RATE) >= FPS(1))
		{

			for (int i = 0; i < 2; ++i)
			{
				MyTracker& tracker = trackers[i];
				tracker.drawTracks();
				/*чтобы добавить задержку, иначе imshow может не отработать*/
				cv::waitKey(1);
				cv::imshow(std::to_string(tracker.m_trackerId), tracker.m_frame);
			}
			
			std::this_thread::sleep_for(FPS(1));
			continue;
		}

		/*Если дошли до сюда, значит со start прошло столько времени, сколько нужно для UPDATE_RATE
		кадров. Выполняем анализ трекером*/
		for (auto& tracker : trackers) 
		{
			tracker.updateTrack();
			tracker.updateBox();

			tracker.initTracker();

			auto& track = tracker.m_track;
			if (tracker.m_track == nullptr)
				continue;

			/*Трекер ищет на новом кадре текущий объект.
			Если не находит, либо если трек долго стоит на месте,
			помечаем его пропавшим. Иначе, добавляем координаты трека
			в список его последних положений.
			Жертвую читаемостью во избежание копипасты, которая была в прошлый раз*/
			if (!tracker.m_pointer->update(tracker.m_frame, track->m_coords) || 
				(track != nullptr && tracker.isStill()))
			{
				track->m_isPresent = false;
				track = nullptr;
			}
			else 
			{
				if (track->m_lastPositions.size() >= STILL_FRAMES)
					track->m_lastPositions.pop_front();
				track->m_lastPositions.push_back({ track->m_coords.x, track->m_coords.y });
			}
		}

		for (int i = 0; i < 2; ++i)
		{
			MyTracker& tracker = trackers[i];
			tracker.drawTracks();
			/*чтобы добавить задержку, иначе imshow может не отработать*/
			cv::waitKey(1);
			cv::imshow(std::to_string(tracker.m_trackerId), tracker.m_frame);
		}
	}
}
