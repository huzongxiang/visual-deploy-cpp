#ifndef OCR_H
#define OCR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <paddle_inference_api.h>
#include "op.h"

class OCRWrapper {
public:
    struct Config {
        std::string modelPath;
        int intraOpNumThreads = 4;  // 默认线程数
        bool useMkldnn = false;
        int recBatchNum = 6;
        int recImgH = 32;
        int recImgW = 320;
        std::string dictPath;
    };

    OCRWrapper(const Config& config);
    ~OCRWrapper();

    std::vector<std::string> infer(const std::vector<cv::Mat>& img_list);

    std::vector<float> infer(const std::vector<cv::Mat>& norm_img_batch, int batch_width, std::vector<int>& predict_shape);

private:
    // 前处理
    void preprocess(const std::vector<cv::Mat>& img_list, std::vector<cv::Mat>& norm_img_batch, int& batch_width);

    // 后处理
    std::vector<std::string> postprocess(const std::vector<float>& predict_batch, const std::vector<int>& predict_shape);

    // 模型相关
    std::shared_ptr<paddle_infer::Predictor> predictor_;
    Config config_;

    std::vector<std::string> labelList_;
    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool isScale_ = true;
    std::vector<int> recImageShape_ = {3, config_.recImgH, config_.recImgW};
    PaddleOCR::CrnnResizeImg resizeOp_;
    PaddleOCR::Normalize normalizeOp_;
    PaddleOCR::PermuteBatch permuteOp_;
};

#endif // OCR_H 