#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/background_segm.hpp>


constexpr int NUM_TRACKERS = 2;
constexpr int UPDATE_RATE = 3;
constexpr int SEARCH_BOX_AREA = 3000;
constexpr int STILL_FRAMES = 10;
constexpr int STILL_RADIUS = 30;
constexpr int LIVE_FRAMES = 240;
constexpr int IOU_THRESHOLD = 0;
constexpr double HIST_THRESHOLD = 0.006;
constexpr int ACTIVATION_FRAMES = 15;

const std::string DEFAULT_PATH1 = "../test.avi";
const std::string DEFAULT_PATH2 = "../test1.avi";

/*Бокс, который потенциально станет треком при соблюдении определенных условий.
Он обводит самое большое изменение в фоне. При этом, изменение не должно быть ниже порога SEARCH_BOX_AREA,
иначе бокс создаваться не будет*/
struct Box
{
	Box(cv::Rect coords) : m_coords(coords) {};
	Box() {};

	cv::Rect m_coords;

	/*Количество кадров, которые живет бокс. Увеличивается всякий раз,
	когда на новом кадре бокс достаточно пересекается с данным. Иначе зануляется*/
	int m_framesAlive = 0;
};


/*Полноценный трек, созданный из бокса*/
struct Track 
{
	Track(const cv::Rect& coords,
		const size_t id,
		const Box& bgBox,
		const size_t trackerId,
		const cv::Mat hist);
	Track() {};

	cv::Rect m_coords;
	int m_id = 0;

	/*Трехмерная гистограмма, необходимая для сравнения похожести.
	Вычисляется значительно медленнее одномерной, но работает заметно точнее.
	Из-за этого всякий раз при появлении трека будут подтормаживания*/
	cv::Mat m_hist;

	/*Вектор координат последних точек, где находился левый верхний угол трека.
	Он пригодится, чтобы удалить трек, стоящий на месте долгое время. Такое происходит,
	когда встроенный трекер opencv постепенно теряет человека и начинает следить за
	статичным фоном*/
	std::list<std::vector<int>> m_lastPositions;

	/*Находится ли трек в кадре*/
	bool m_isPresent = true;

	/*Оставшееся время жизни, в кадрах*/
	int m_liveFrames = LIVE_FRAMES;

	/*Становится true только когда время жизни истекло*/
	bool m_expired = false;
	int m_trackerId = 0;
};

class MyTracker
{
private:
	cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgSub = cv::createBackgroundSubtractorMOG2(500, 150, false);
	cv::Mat m_fgMask = cv::Mat();
	Box m_bgBox;
	std::vector<std::shared_ptr<Track>>& m_trackList;


public:
	MyTracker(int i, std::vector<std::shared_ptr<Track>>& trackList, cv::Mat& frame) :
		m_trackerId(i), m_trackList(trackList), m_frame(frame) {};

	/*Все указатели среди членов класса сделал интеллектуальными, поэтому не чищу память явно
	в деструкторе. Автоматически должны вызваться деструкторы этих указателей, которые почистят память*/
	~MyTracker() {};

	int m_trackerId;
	cv::Mat& m_frame;
	cv::Ptr<cv::TrackerKCF> m_pointer = cv::TrackerKCF::create();

	/*Указывает на активный трек. Shared вместо unique, т.к. этот
	указатель идентичен одному из тех, которые лежат в trackList.
	То есть, копия не единственная*/
	std::shared_ptr<Track> m_track = nullptr;

	/*Ищет изменения в фоне и возвращает бокс, который обводит
	самое большое изменение. Если оно слишком мало, то бокс нулевой*/
	Box searchBox();

	/*Подсчет трехмерной гистограммы внутри бокса*/
	cv::Mat calcBoxHist() const;

	/*Сравнение гистограмм треков*/
	double compHist(const std::shared_ptr<Track> compTrack) const;

	/*Инициализация трекера*/
	void initTracker();

	/*Обновляет время оставшейся жизни отсутствующих в кадре треков.
	Если время заканчивается, помечает трек как пропавший*/
	void updateTrack(); 

	/*Обновляет время существования бокса.
	Бокс достаточно пересекается с активным треком => зануляем бокс
	Бокс не пересекается с треком, но пересекается с боксом => увеличиваем время
	Бокс нулевой => зануляем время
	Если на данный момент нет активного трека, а также бокс пересекается с боксом
	из прошлого кадра => увеличиваем время*/
	void updateBox();

	/*Проверяем, стоит ли трек на месте достаточно долго. Проверка идет только когда
	трек активен*/
	bool isStill() const;

	/*Площадь пересечения прямоугольников, деленная на площадь их объединения*/
	double IOU(const cv::Rect& rect1, const cv::Rect& rect2) const;

	/*Отображение треков и их id*/
	void drawTracks();
};

