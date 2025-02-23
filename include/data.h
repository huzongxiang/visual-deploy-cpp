#ifndef DATA_H
#define DATA_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <codecvt>
#include <ctime>
#include <iostream>
#include "yolo_wrapper.h"

namespace data_utils {
    // 计算向量元素乘积
    size_t vectorProduct(const std::vector<int64_t> &vector);

    // 字符转换为宽字符串
    std::wstring charToWstring(const char *str);

    // 加载类别名称
    void loadNames(const std::vector<std::string>& classNames);

    // 可视化检测结果
    void visualizeDetection(cv::Mat &im, std::vector<Detection> &results,
                           const std::vector<std::string> &classNames);

    // letterbox 图像处理
    void letterbox(const cv::Mat &image, cv::Mat &outImage,
                  const cv::Size &newShape,
                  const cv::Scalar &color,
                  bool auto_,
                  bool scaleFill,
                  bool scaleUp,
                  int stride);

    // 坐标缩放
    void scaleCoords(cv::Rect &coords, const cv::Size &imageShape, const cv::Size &imageOriginalShape);

    // 数值裁剪
    template <typename T>
    T clip(const T &n, const T &lower, const T &upper);

    // 颜色向量
    extern std::vector<cv::Scalar> colors;
}

#endif // DATA_H 