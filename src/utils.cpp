#include "utils.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <map>

using namespace cv;
using namespace std;

// 初始化状态变量
TrafficSignalStatus currentStatus = TrafficSignalStatus::Normal;

std::vector<std::string> ReadDict(const std::string &path) noexcept {
    std::vector<std::string> m_vec;
    std::ifstream in(path);
    if (in) {
        for (;;) {
            std::string line;
            if (!getline(in, line))
                break;
            m_vec.emplace_back(std::move(line));
        }
    } else {
        std::cout << "no such label file: " << path << ", exit the program..."
                  << std::endl;
        exit(1);
    }
    return m_vec;
}

std::string parseCommandLineArgs(int argc, char* argv[], const std::string& defaultPath) {
    string configPath = defaultPath;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--cfg" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--cfg") {
            cerr << "错误：--cfg 需要指定配置文件路径" << endl;
            exit(-1);
        }
    }
    cout << "使用的配置文件: " << configPath << endl;
    return configPath;
}

bool loadConfig(const string& configPath, 
               string& videoSource,
               YOLOWrapper::Config& yoloConfig,
               OCRWrapper::Config& ocrConfig,
               Rect& signalLightROI,
               Rect& timerROI,
               bool& enableVideoOutput,
               string& videoOutputPath,
               string& videoCodec,
               int& videoFps) {
    try {
        YAML::Node config = YAML::LoadFile(configPath);

        videoSource = config["video_source"].as<string>();

        // 加载 YOLO 配置
        auto yoloNode = config["yolo_config"];
        yoloConfig.modelPath = yoloNode["model_path"].as<string>();
        yoloConfig.confThreshold = yoloNode["conf_threshold"].as<float>();
        yoloConfig.iouThreshold = yoloNode["iou_threshold"].as<float>();
        yoloConfig.intraOpNumThreads = yoloNode["num_threads"].as<int>();

        // 加载 OCR 配置
        auto ocrNode = config["ocr_config"];
        ocrConfig.modelPath = ocrNode["model_dir"].as<string>();
        ocrConfig.intraOpNumThreads = ocrNode["intra_op_num_threads"].as<int>();
        ocrConfig.useMkldnn = ocrNode["use_mkldnn"].as<bool>();
        ocrConfig.recBatchNum = ocrNode["rec_batch_num"].as<int>();
        ocrConfig.recImgH = ocrNode["rec_img_h"].as<int>();
        ocrConfig.recImgW = ocrNode["rec_img_w"].as<int>();
        ocrConfig.dictPath = ocrNode["dict_path"].as<string>();
        
        signalLightROI.x = config["signalLightROI"]["x"].as<int>();
        signalLightROI.y = config["signalLightROI"]["y"].as<int>();
        signalLightROI.width = config["signalLightROI"]["width"].as<int>();
        signalLightROI.height = config["signalLightROI"]["height"].as<int>();

        timerROI.x = config["timerROI"]["x"].as<int>();
        timerROI.y = config["timerROI"]["y"].as<int>();
        timerROI.width = config["timerROI"]["width"].as<int>();
        timerROI.height = config["timerROI"]["height"].as<int>();

        enableVideoOutput = config["video_output"]["enable"].as<bool>();
        videoOutputPath = config["video_output"]["path"].as<string>();
        videoCodec = config["video_output"]["codec"].as<string>();
        videoFps = config["video_output"]["fps"].as<int>();

        return true;
    } catch (const YAML::Exception& e) {
        cerr << "配置文件加载失败: " << e.what() << endl;
        return false;
    }
}

// 实现 argsort 函数
std::vector<size_t> argsort(const std::vector<float>& array) noexcept {
    std::vector<size_t> array_index(array.size(), 0);
    for (size_t i = 0; i < array.size(); ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](size_t pos1, size_t pos2) noexcept {
                return (array[pos1] < array[pos2]);
              });

    return array_index;
}

void drawTimerInfo(cv::Mat& frame, 
                  const cv::Rect& timerROI,
                  const std::string& timerValue) {
    // 绘制计时区域框
    if (timerROI.area() > 0) {
        cv::rectangle(frame, timerROI, cv::Scalar(0, 255, 0), 2);
    }

    // 显示计时结果
    if (!timerValue.empty()) {
        int baseLine;
        cv::Size textSize = cv::getTextSize(timerValue, cv::FONT_HERSHEY_SIMPLEX, 1.5, 2, &baseLine);
        cv::Point textOrg(
            timerROI.x + (timerROI.width - textSize.width)/2,  // 水平居中
            timerROI.y + timerROI.height + textSize.height + 10  // 框下方10像素
        );
        
        cv::putText(frame, timerValue, 
                    textOrg,
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    }
}

void drawStatusInfo(cv::Mat& frame, 
                   TrafficSignalStatus status,
                   const std::string& colorName,
                   const std::string& timerValue) {
    std::string statusText;
    cv::Scalar color;

    // 中英颜色名称映射
    std::map<std::string, std::string> colorMap = {
        {"绿", "Green"},
        {"红", "Red"},
        {"黄", "Yellow"}
    };
    std::string englishColor = colorMap.count(colorName) ? colorMap[colorName] : "Unknown";

    switch (status) {
        case TrafficSignalStatus::Normal:
            statusText = " Status: [Normal] " + englishColor + " Light | Time Left: " + timerValue + "s";
            color = cv::Scalar(0, 255, 0); // 绿色
            std::cout << "状态: 正常 - " << colorName << "灯亮 剩余时间: " << timerValue << "秒" << std::endl;
            break;
        case TrafficSignalStatus::SignalMissing:
            statusText = " Status: [Alert] No Traffic Light Detected";
            color = cv::Scalar(0, 0, 255); // 红色
            std::cout << "状态: [警告] - 未检测到信号灯，信号灯工作异常" << std::endl;
            break;
        case TrafficSignalStatus::TimerMissing:
            statusText = " Status: [Alert] Light On But No Timer";
            color = cv::Scalar(0, 255, 255); // 黄色
            std::cout << "状态: [警告] - 检测到信号灯，未检测到计时数字，信号灯超时" << std::endl;
            break;
    }

    std::cout << "绘制状态信息: " << statusText << std::endl;
    // 绘制文字
    int baseline;
    cv::Size textSize = cv::getTextSize(statusText, cv::FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseline);
    cv::putText(frame, statusText,
                cv::Point(30, 50),  // 调整到更显眼的位置
                cv::FONT_HERSHEY_SIMPLEX, 
                1.2,  // 增大字体
                color, 
                2);   // 加粗文字
}
