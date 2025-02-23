#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class YOLOWrapper {
public:
    struct Config {
        std::string modelPath;
        float confThreshold = 0.5;
        float iouThreshold = 0.45;
        int intraOpNumThreads = 4;
    };

    YOLOWrapper(const Config& config);
    ~YOLOWrapper();
    
    std::vector<Detection> infer(cv::Mat& frame);

private:
    // ONNX Runtime 相关
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    
    // 模型元数据
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;
    
    // 配置参数
    Config config_;
    bool isDynamicInputShape_ = false;
    
    // 预处理
    void preprocess(cv::Mat& image, float*& blob, std::vector<int64_t>& tensorShape);
    
    // 后处理
    std::vector<Detection> postprocess(
        const cv::Size& resizedImageShape,
        const cv::Size& originalImageShape,
        std::vector<Ort::Value>& outputTensors
    );

    // 获取最佳类别信息
    void getBestClassInfo(std::vector<float>::iterator it, float& bestConf, int& bestClassId);
}; 