#include "ocr.h"
#include "paddle_inference_api.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "utils.h"

using namespace paddle_infer;
using namespace std;

OCRWrapper::OCRWrapper(const Config& config) : config_(config) {
    try {
        // 初始化配置
        paddle_infer::Config paddleConfig;
        paddleConfig.SetModel(config.modelPath + "/inference.pdmodel", 
                             config.modelPath + "/inference.pdiparams");
        paddleConfig.EnableMKLDNN();  // 启用 MKLDNN 加速
        paddleConfig.SetCpuMathLibraryNumThreads(config.intraOpNumThreads);
        paddleConfig.EnableMemoryOptim();  // 启用内存优化

        // 创建推理器
        predictor_ = paddle_infer::CreatePredictor(paddleConfig);
        if (!predictor_) {
            throw runtime_error("Failed to create predictor");
        }
    } catch (const std::exception& e) {
        cerr << "OCR初始化失败: " << e.what() << endl;
        predictor_ = nullptr;
    }

    // 初始化其他成员变量
    labelList_ = ReadDict(config.dictPath);
    labelList_.emplace(labelList_.begin(), "#"); // blank char for ctc
    labelList_.emplace_back(" ");
    recImageShape_ = {3, config.recImgH, config.recImgW};
}

OCRWrapper::~OCRWrapper() {
    // 清理资源
}

void OCRWrapper::preprocess(const std::vector<cv::Mat>& img_list, std::vector<cv::Mat>& norm_img_batch, int& batch_width) {
    size_t img_num = img_list.size();
    std::vector<float> width_list;
    for (size_t i = 0; i < img_num; ++i) {
        width_list.emplace_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<size_t> indices = std::move(argsort(width_list));

    int imgH = this->recImageShape_[1];
    int imgW = this->recImageShape_[2];
    float max_wh_ratio = imgW * 1.0 / imgH;
    for (size_t ino = 0; ino < img_num; ++ino) {
        int h = img_list[indices[ino]].rows;
        int w = img_list[indices[ino]].cols;
        float wh_ratio = w * 1.0 / h;
        max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
    }

    batch_width = imgW;
    for (size_t ino = 0; ino < img_num; ++ino) {
        cv::Mat srcimg;
        img_list[indices[ino]].copyTo(srcimg);
        cv::Mat resize_img;
        this->resizeOp_.Run(srcimg, resize_img, max_wh_ratio,
                             false, this->recImageShape_);
        this->normalizeOp_.Run(resize_img, this->mean_, this->scale_,
                                this->isScale_);
        batch_width = std::max(resize_img.cols, batch_width);
        norm_img_batch.emplace_back(std::move(resize_img));
    }
}

std::vector<float> OCRWrapper::infer(const std::vector<cv::Mat>& norm_img_batch, int batch_width, std::vector<int>& predict_shape) {
    int batch_num = norm_img_batch.size();
    std::vector<float> input(batch_num * 3 * this->recImageShape_[1] * batch_width, 0.0f);
    this->permuteOp_.Run(norm_img_batch, input.data());

    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape(std::vector<int>{batch_num, 3, this->recImageShape_[1], batch_width});
    input_t->CopyFromCpu(input.data());

    try {
        this->predictor_->Run();
    } catch (const std::exception& e) {
        cerr << "OCR推理失败: " << e.what() << endl;
        return {};
    }

    std::vector<float> predict_batch;
    auto output_names = this->predictor_->GetOutputNames();
    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    predict_shape = output_t->shape();

    size_t out_num = std::accumulate(predict_shape.begin(), predict_shape.end(),
                                     1, std::multiplies<int>());
    predict_batch.resize(out_num);
    output_t->CopyToCpu(predict_batch.data());
    
    return predict_batch;
}

std::vector<std::string> OCRWrapper::postprocess(const std::vector<float>& predict_batch, const std::vector<int>& predict_shape) {
    std::vector<std::string> results;
    int batch_size = predict_shape[0];
    int imgW = predict_shape[1];
    int num_classes = predict_shape[2];  // 类别数，即 labelList_.size()
    for (int m = 0; m < batch_size; ++m) {
        std::string str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;

        for (int n = 0; n < imgW; ++n) {
            // 获取当前时间步的最大值索引
            argmax_idx = int(std::distance(
                &predict_batch[(m * imgW + n) * num_classes],
                std::max_element(&predict_batch[(m * imgW + n) * num_classes],
                                 &predict_batch[(m * imgW + n + 1) * num_classes])));
            // 获取最大值
            max_value = float(*std::max_element(
                &predict_batch[(m * imgW + n) * num_classes],
                &predict_batch[(m * imgW + n + 1) * num_classes]));

            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                str_res += labelList_[argmax_idx];
            }
            last_index = argmax_idx;
        }
        score /= count;
        if (std::isnan(score)) {
            continue;
        }
        results.push_back(str_res);
    }
    return results;
}

std::vector<std::string> OCRWrapper::infer(const std::vector<cv::Mat>& img_list) {
    std::vector<cv::Mat> norm_img_batch;
    int batch_width = 0;
    preprocess(img_list, norm_img_batch, batch_width);

    std::vector<int> predict_shape;
    std::vector<float> predict_batch = infer(norm_img_batch, batch_width, predict_shape);

    return postprocess(predict_batch, predict_shape);
}