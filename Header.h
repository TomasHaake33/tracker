#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <list>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <buffers.h>


const std::string VIDEO_PATH = "../test.avi";
const char MODEL_PATH[] = "../GeneralNMHuman_v1.0GPU_onnx.onnx";
const char ENGINE_PATH[] = "../Model.engine";

constexpr int UPDATE_RATE = 1;
constexpr int ACTIVATION_FRAMES = 20;
constexpr int LIVE_FRAMES = 80;


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

/*Класс, который использую для работы с нейросетью*/
class NNet {
private:
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Logger m_logger;
    samplesCommon::ManagedBuffer m_inputBuff;
    samplesCommon::ManagedBuffer m_outputBuff;

public:
    NNet() {};

    /*Чистить нечего, поэтому пустой*/
    ~NNet() {};

    /*Считывает из modelPath модель .onnx, создает из нее движок .engine
    и сохраняет его в файл*/
    bool buildEngine(const char modelPath[]);

    /*Загружает из файла движок*/
    bool load();

    /*Смотрит на размеры, входа и выхода, которые ожидает нейросеть.
    Задает их соответственно и выполняет инференс на видеокарте,
    записывает результат в одномерный вектор. Форматировать в более удобный
    формат буду позже*/
    bool infer(const cv::Mat& img, std::vector<float>& features);
};

struct Track {

    Track(const cv::Rect& box, const int id = 0) : m_box(box), m_id(id) {};
    Track() {};

    //Координаты бокса трека
    cv::Rect m_box;
    //Кадры активации. Когда становится достаточно большим, трек активируется
    int m_actvFrames = 0;
    //Оставшееся время жизни в кадрах
    int m_liveFrames = LIVE_FRAMES;
    bool m_activated = false;
    //Находится ли трек в кадре
    bool m_present = false;
    //Считается ли трек исчезнувшим
    bool m_expired = false;
    int m_id = 0;

};

class MyTracker
{
private:
    //Можно было public наследование сделать, но мне кажется так логичней
    NNet m_model;

    //Векторы для обработки результатов инференса
    std::vector<float> m_rawOutputs;
    std::vector<cv::Rect> m_rects;
    std::vector<float> m_scores;
    std::vector<cv::Rect> m_outRects;

    //Все существовавшие треки
    std::vector<std::unique_ptr<Track>> m_tracks;

public:
    MyTracker() {};

    /*Всё почистится автоматически, оставляю пустым деструктор*/
    ~MyTracker() {};

    /*Обновляет время жизни треков и создает новые*/
    void updateTracks();

    /*Вспомогательная функция для updateTracks. Ищет индексы выходов в
    массиве с ними, которые не пересекаются ни с какими треками. Проще говоря,
    новые треки*/
    std::vector<int> searchNew(const std::vector<cv::Rect>& outputs) const;
    bool drawTracks(cv::Mat& frame);

    double IOU(const cv::Rect& rect1, const cv::Rect& rect2) const;

    /*Из сырого вектора выходов сети ищет те, которые проходят по порогу вероятности,
    записывает их в private члены. Еще делает ресайз координат под размеры видео, т.к.
    они изначально нормализованы*/
    void processOutputs(const double thresh, const cv::Mat& frame);

    /*Упрощенный вариант nms. Сначала отсеивает скопления боксов, где недостаточно
    "соседей". Если соседствующих боксов много, выделяет из них тот, у которого
    наибольший score. Добавляет такие боксы в private член*/
    void nms(double thresh, int neighbors);

    void inferModel(cv::Mat& blob) { m_model.infer(blob, m_rawOutputs); };
    void loadModel() { m_model.load(); };
    void buildEngine(const char modelPath[]) { m_model.buildEngine(modelPath); };

    //Чистит private члены, иначе будут скапливаться результаты инференсов
    void clearOutputs();


};