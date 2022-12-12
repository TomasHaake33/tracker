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

/*�����, ������� ��������� ��� ������ � ����������*/
class NNet {
private:
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Logger m_logger;
    samplesCommon::ManagedBuffer m_inputBuff;
    samplesCommon::ManagedBuffer m_outputBuff;

public:
    NNet() {};

    /*������� ������, ������� ������*/
    ~NNet() {};

    /*��������� �� modelPath ������ .onnx, ������� �� ��� ������ .engine
    � ��������� ��� � ����*/
    bool buildEngine(const char modelPath[]);

    /*��������� �� ����� ������*/
    bool load();

    /*������� �� �������, ����� � ������, ������� ������� ���������.
    ������ �� �������������� � ��������� �������� �� ����������,
    ���������� ��������� � ���������� ������. ������������� � ����� �������
    ������ ���� �����*/
    bool infer(const cv::Mat& img, std::vector<float>& features);
};

struct Track {

    Track(const cv::Rect& box, const int id = 0) : m_box(box), m_id(id) {};
    Track() {};

    //���������� ����� �����
    cv::Rect m_box;
    //����� ���������. ����� ���������� ���������� �������, ���� ������������
    int m_actvFrames = 0;
    //���������� ����� ����� � ������
    int m_liveFrames = LIVE_FRAMES;
    bool m_activated = false;
    //��������� �� ���� � �����
    bool m_present = false;
    //��������� �� ���� �����������
    bool m_expired = false;
    int m_id = 0;

};

class MyTracker
{
private:
    //����� ���� public ������������ �������, �� ��� ������� ��� ��������
    NNet m_model;

    //������� ��� ��������� ����������� ���������
    std::vector<float> m_rawOutputs;
    std::vector<cv::Rect> m_rects;
    std::vector<float> m_scores;
    std::vector<cv::Rect> m_outRects;

    //��� �������������� �����
    std::vector<std::unique_ptr<Track>> m_tracks;

public:
    MyTracker() {};

    /*�� ���������� �������������, �������� ������ ����������*/
    ~MyTracker() {};

    /*��������� ����� ����� ������ � ������� �����*/
    void updateTracks();

    /*��������������� ������� ��� updateTracks. ���� ������� ������� �
    ������� � ����, ������� �� ������������ �� � ������ �������. ����� ������,
    ����� �����*/
    std::vector<int> searchNew(const std::vector<cv::Rect>& outputs) const;
    bool drawTracks(cv::Mat& frame);

    double IOU(const cv::Rect& rect1, const cv::Rect& rect2) const;

    /*�� ������ ������� ������� ���� ���� ��, ������� �������� �� ������ �����������,
    ���������� �� � private �����. ��� ������ ������ ��������� ��� ������� �����, �.�.
    ��� ���������� �������������*/
    void processOutputs(const double thresh, const cv::Mat& frame);

    /*���������� ������� nms. ������� ��������� ��������� ������, ��� ������������
    "�������". ���� ������������� ������ �����, �������� �� ��� ���, � ��������
    ���������� score. ��������� ����� ����� � private ����*/
    void nms(double thresh, int neighbors);

    void inferModel(cv::Mat& blob) { m_model.infer(blob, m_rawOutputs); };
    void loadModel() { m_model.load(); };
    void buildEngine(const char modelPath[]) { m_model.buildEngine(modelPath); };

    //������ private �����, ����� ����� ������������ ���������� ����������
    void clearOutputs();


};