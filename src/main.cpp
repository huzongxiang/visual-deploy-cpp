#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "yolo_wrapper.h"
#include "ocr.h"
#include "utils.h"
#include "data.h"
#include <sys/stat.h>

using namespace cv;
using namespace std;

// 提供接口获取状态
TrafficSignalStatus getTrafficSignalStatus() {
    return currentStatus;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数并获取配置路径
    string configPath = parseCommandLineArgs(argc, argv, "../config.yaml");
    
    // 定义配置变量
    string videoSource;
    YOLOWrapper::Config yoloConfig;
    OCRWrapper::Config ocrConfig;
    Rect signalLightROI, timerROI;
    bool enableVideoOutput;
    string videoOutputPath;
    string videoCodec;
    int videoFps;

    // 加载配置文件
    if (!loadConfig(configPath, videoSource, yoloConfig, ocrConfig, signalLightROI, timerROI, 
                    enableVideoOutput, videoOutputPath, videoCodec, videoFps)) {
        return -1;
    }

    // 打开视频流或摄像头
    VideoCapture cap(videoSource);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open the video stream!" << endl;
        return -1;
    }

    // 调试信息用于测试，创建保存帧的目录
    string frameOutputDir = "../results/output_frames";
    if (mkdir(frameOutputDir.c_str(), 0777) != 0 && errno != EEXIST) {
        cerr << "Error: Could not create output directory " << frameOutputDir << endl;
        return -1;
    }
    int frameCounter = 0;  // 帧计数器

    // 获取视频的帧率和分辨率
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = videoFps;  // 如果无法获取帧率，使用配置中的默认值

    // 创建 VideoWriter 对象，用于保存视频
    VideoWriter videoWriter;
    if (enableVideoOutput) {
        int fourcc = VideoWriter::fourcc(videoCodec[0], videoCodec[1], videoCodec[2], videoCodec[3]);
        videoWriter.open(videoOutputPath, fourcc, fps, Size(frameWidth, frameHeight));
        if (!videoWriter.isOpened()) {
            cerr << "Error: Could not open the output video file for write." << endl;
        } else {
            cout << "视频输出已启用，保存路径: " << videoOutputPath << endl;
        }
    }

    // 定义连续异常计数器和阈值
    int signalAnomalyCounter = 0;    // 信号灯异常计数器
    int ocrAnomalyCounter = 0;       // OCR异常计数器
    const int signalAnomalyThreshold = 30;  // 信号灯异常阈值（3秒，假设10fps）
    const int ocrAnomalyThreshold = 60;    // OCR异常阈值（6秒）

    // 初始化 YOLO 模型
    YOLOWrapper yoloWrapper(yoloConfig);

    // 初始化颜色（必须在第一次调用visualizeDetection之前）
    std::vector<std::string> classNames = {"Green", "Red", "Yellow"};
    data_utils::loadNames(classNames);

    // 初始化 OCR 模型
    OCRWrapper ocrWrapper(ocrConfig);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // 如果定义了信号灯区域，则进行裁切
        Mat signalLightFrame = frame;
        if (signalLightROI.area() > 0) {
            signalLightFrame = frame(signalLightROI);
        }

        // 使用 YOLOWrapper 进行推理
        vector<Detection> detections = yoloWrapper.infer(signalLightFrame);
        cout << "检测到目标数量: " << detections.size() << endl;
        
        // 如果信号灯区域进行了裁切，则将检测结果映射回原始帧坐标系
        if (signalLightROI.area() > 0) {
            for (auto &detection : detections) {
                detection.box.x += signalLightROI.x;
                detection.box.y += signalLightROI.y;
            }
        }
        
        // 处理检测结果
        vector<Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;
        bool hasDetection = false;
        if (!detections.empty()) {
            for (const auto& detection : detections) {
                boxes.push_back(detection.box);
                classIds.push_back(detection.classId);
                confidences.push_back(detection.confidence);
            }
            hasDetection = true;
            cout << "hasDetection: " << hasDetection << endl;
        }

        // 如果没有检测到目标，直接累计异常并跳过后续处理
        if (!hasDetection) {
            signalAnomalyCounter++;
            if (signalAnomalyCounter >= signalAnomalyThreshold) {
                cout << "[报警] 连续 " << signalAnomalyCounter << " 帧未检测到信号灯" << endl;
                currentStatus = TrafficSignalStatus::SignalMissing;
                signalAnomalyCounter = 0;
            }
            // continue;  // 跳过后续处理
        } else {
            signalAnomalyCounter = 0;  // 信号灯正常时重置
            currentStatus = TrafficSignalStatus::Normal;
        }

        // 如果定义了计时区域，则进行裁切
        Mat timerFrame = frame;
        if (timerROI.area() > 0) {
            timerFrame = frame(timerROI);
        }

        // 执行 OCR，检测计时数字
        vector<Mat> timerFrames = {timerFrame};
        vector<string> ocrResults = ocrWrapper.infer(timerFrames);
        cout << "ocrResults: " << ocrResults.size() << endl;
        string timerValue = ocrResults.empty() ? "" : ocrResults[0];
        cout << "timerValue: " << timerValue << endl;
        bool timerAnomaly = timerValue.empty();
        
        if (timerAnomaly) {
            ocrAnomalyCounter++;
            // 仅在当前状态正常时更新为OCR异常
            if (currentStatus == TrafficSignalStatus::Normal) {
                currentStatus = TrafficSignalStatus::TimerMissing;
            }
            cout << "timerAnomaly: " << timerAnomaly << endl;
        } else {
            ocrAnomalyCounter = 0;
            currentStatus = TrafficSignalStatus::Normal;
            cout << "timerAnomaly: " << timerAnomaly << endl;
        }

        // 处理正常输出：例如显示颜色及剩余时间（此处仅作为示例，无异常时才显示）
        int signalClassId = -1;
        if (!boxes.empty() && !classIds.empty()) {
            // 取第一个检测结果
            signalClassId = classIds[0];
        }
        string colorName;
        switch(signalClassId) {
            case 0: colorName = "绿"; break;
            case 1: colorName = "红"; break;
            case 2: colorName = "黄"; break;
        }

        if (!timerAnomaly) {
            cout << "[正常] " << colorName << "灯亮 剩余时间: " << timerValue << "秒" << endl;
        }

        // 绘制状态信息
        drawStatusInfo(frame, currentStatus, colorName, timerValue);

        // 可视化部分调整
        drawTimerInfo(frame, timerROI, timerValue);

        // 使用通用可视化函数
        data_utils::visualizeDetection(frame, detections, classNames);

        // 调试信息用于测试，保存当前帧为图像文件，正式请删除
        string framePath = format("%s/frame_%04d.jpg", frameOutputDir.c_str(), frameCounter++);
        imwrite(framePath, frame);

        // 如果启用了视频输出，将当前帧写入视频文件
        if (enableVideoOutput && videoWriter.isOpened()) {
            videoWriter.write(frame);
        }

        if (waitKey(1) == 27) break;  // 按下 ESC 键退出
    }

    cap.release();
    if (videoWriter.isOpened()) {
        videoWriter.release();
        cout << "视频保存完成。" << endl;
    }
    destroyAllWindows();
    return 0;
}