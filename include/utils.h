#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include "yolo_wrapper.h"
#include "ocr.h"

// 定义状态变量
enum class TrafficSignalStatus {
    Normal,          // 正常
    SignalMissing,   // 未检测到信号灯
    TimerMissing     // 未检测到计时数字
};

extern TrafficSignalStatus currentStatus;  // 当前状态

// 命令行参数解析
std::string parseCommandLineArgs(int argc, char* argv[], const std::string& defaultPath);

// 配置文件加载
bool loadConfig(const std::string& configPath, 
               std::string& videoSource,
               YOLOWrapper::Config& yoloConfig,
               OCRWrapper::Config& ocrConfig,
               cv::Rect& signalLightROI,
               cv::Rect& timerROI,
               bool& enableVideoOutput,
               std::string& videoOutputPath,
               std::string& videoCodec,
               int& videoFps);

// 读取标签文件
std::vector<std::string> ReadDict(const std::string &path) noexcept;

// 实现 argsort 函数
std::vector<size_t> argsort(const std::vector<float>& array) noexcept;

// 绘制计时区域和结果
void drawTimerInfo(cv::Mat& frame, 
                  const cv::Rect& timerROI,
                  const std::string& timerValue);

// 绘制状态信息
void drawStatusInfo(cv::Mat& frame, 
                   TrafficSignalStatus status,
                   const std::string& colorName = "",
                   const std::string& timerValue = "");

#endif // UTILS_H
