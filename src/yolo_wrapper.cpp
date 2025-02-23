#include "yolo_wrapper.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include "data.h"

using namespace cv;
using namespace Ort;
using namespace std;

// 初始化 ONNX Runtime 模型
YOLOWrapper::YOLOWrapper(const Config& config) 
    : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-ONNXRuntime") {
    
    sessionOptions_.SetIntraOpNumThreads(config.intraOpNumThreads);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 创建 session
    session_ = std::make_unique<Ort::Session>(env_, config.modelPath.c_str(), sessionOptions_);

    // 获取输入和输出节点数量
    const size_t num_input_nodes = session_->GetInputCount();
    const size_t num_output_nodes = session_->GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    // 处理输入节点
    for (size_t i = 0; i < num_input_nodes; i++) {
        // 获取输入节点名称
        inputNames_.push_back(session_->GetInputName(i, allocator));

        // 获取输入形状信息
        Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(i);
        auto tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = tensor_info.GetShape();

        // 检查是否为动态输入形状
        isDynamicInputShape_ = false;
        if (inputShape_[2] == -1 && inputShape_[3] == -1) {
            std::cout << "Dynamic input shape detected" << std::endl;
            isDynamicInputShape_ = true;
        }
    }

    // 处理输出节点
    for (size_t i = 0; i < num_output_nodes; i++) {
        // 获取输出节点名称
        outputNames_.push_back(session_->GetOutputName(i, allocator));

        // 获取输出形状信息
        Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(i);
        auto tensor_info = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape_ = tensor_info.GetShape();
    }
}

YOLOWrapper::~YOLOWrapper() {
    // 清理输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    for (const char* name : inputNames_) {
        allocator.Free(const_cast<void*>(static_cast<const void*>(name)));
    }
    for (const char* name : outputNames_) {
        allocator.Free(const_cast<void*>(static_cast<const void*>(name)));
    }
}

std::vector<Detection> YOLOWrapper::infer(cv::Mat& frame) {
    float* blob = nullptr;
    std::vector<int64_t> inputTensorShape = {1, 3, -1, -1}; // 动态输入形状
    preprocess(frame, blob, inputTensorShape);

    size_t inputTensorSize = data_utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));

    std::vector<Ort::Value> outputTensors = session_->Run(Ort::RunOptions{nullptr},
                                                          inputNames_.data(),
                                                          inputTensors.data(),
                                                          1,
                                                          outputNames_.data(),
                                                          outputNames_.size());

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> results = postprocess(resizedShape, frame.size(), outputTensors);

    delete[] blob;

    return results;
}

// 预处理函数
void YOLOWrapper::preprocess(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape) {
    cv::Mat resizedImage, floatImage;
    // BGR -> RGB
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);

    // letterbox 缩放，保持长宽比
    data_utils::letterbox(resizedImage, resizedImage, 
                    cv::Size((int)this->inputShape_[2], (int)this->inputShape_[3]),  // 使用输入张量的尺寸
                    cv::Scalar(114, 114, 114),  // 填充颜色
                    false,  // isDynamicInputShape
                    false,  // scaleFill
                    true,   // scaleUp
                    32);    // stride

    // 更新输入张量形状
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // 归一化到 [0,1]
    resizedImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    // 分配内存
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    // HWC -> CHW 转换
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

// 后处理函数
std::vector<Detection> YOLOWrapper::postprocess(
    const cv::Size& resizedImageShape,
    const cv::Size& originalImageShape,
    std::vector<Ort::Value>& outputTensors
) {
    // 解析输出数据
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float* boxOutput = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat output0 = cv::Mat(cv::Size((int)outputShape_[2], (int)outputShape_[1]), CV_32F, boxOutput).t();
    float* output0ptr = (float*)output0.data;
    int rows = (int)outputShape_[2];
    int cols = (int)outputShape_[1];

    for (int i = 0; i < rows; i++) {
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        float confidence;
        int classId;
        
        getBestClassInfo(it.begin(), confidence, classId);

        if (confidence > config_.confThreshold) {
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            
            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, config_.confThreshold, config_.iouThreshold, indices);

    std::vector<Detection> results;
    for (int idx : indices) {
        Detection res;
        res.box = cv::Rect(boxes[idx]);
        res.confidence = confs[idx];
        res.classId = classIds[idx];

        data_utils::scaleCoords(res.box, resizedImageShape, originalImageShape);

        results.emplace_back(res);
    }

    return results;
}

void YOLOWrapper::getBestClassInfo(std::vector<float>::iterator it, float& bestConf, int& bestClassId) {
    // 跳过前4个坐标值
    it += 4;
    
    // 找到最大置信度的类别
    bestConf = 0;
    bestClassId = 0;
    for (int i = 0; i < 3; i++) {
        if (it[i] > bestConf) {
            bestConf = it[i];
            bestClassId = i;
        }
    }
}
