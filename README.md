# 交通信号灯识别系统

## 项目简介
该项目是一个基于计算机视觉的交通信号灯识别系统，能够实时检测交通信号灯的颜色状态和倒计时数字。系统集成了YOLO目标检测和OCR文字识别两个主要功能模块，可以准确识别红、黄、绿三种信号灯状态，并读取倒计时显示器上的数字。

## 系统依赖
- OpenCV
- ONNXRuntime (v1.11.1)
- PaddleInference
- yaml-cpp
- C++14 或更高版本
- CMake 3.10 或更高版本

## 目录结构
```
project/
├── deps/                          # 依赖库目录
│   ├── onnxruntime-linux-x64-1.11.1/
│   ├── paddle_inference/
│   └── dict/                     # OCR字典文件
├── include/                      # 头文件目录
│   ├── yolo_wrapper.h           # YOLO检测器封装
│   ├── ocr.h                    # OCR识别器封装
│   ├── utils.h                  # 工具函数
│   ├── data.h                   # 数据处理相关
│   └── op.h                     # 图像操作相关
├── src/                         # 源文件目录
│   ├── main.cpp                 # 主程序
│   ├── yolo_wrapper.cpp
│   ├── ocr.cpp
│   ├── utils.cpp
│   ├── data.cpp
│   └── op.cpp
├── models/                      # 模型文件目录
│   ├── YOLO/                   # YOLO模型
│   └── ch_PP-OCRv3_rec_infer/ # OCR模型
├── config.yaml                  # 配置文件
└── CMakeLists.txt              # CMake构建文件
```

## 功能特点
- 实时检测交通信号灯状态（红、黄、绿）
- 识别倒计时数字
- 支持视频流和摄像头输入
- 异常状态检测和提示
- 可视化检测结果
- 支持视频保存功能

## 编译说明
1. 确保已安装所需依赖：
   - OpenCV
   - yaml-cpp

2. 准备第三方库：
   - 将ONNXRuntime库放置在 `deps/onnxruntime-linux-x64-1.11.1/` 目录
   - 将PaddleInference库放置在 `deps/paddle_inference/` 目录

3. 编译项目：
```bash
mkdir build && cd build
cmake ..
make
```

## 配置说明
在 `config.yaml` 中配置以下参数：

### 视频源配置
```yaml
video_source: /path/to/video  # 视频文件路径或摄像头索引(0)
```

### YOLO模型配置
```yaml
yolo_config:
  model_path: "/path/to/yolo/model"
  num_threads: 4
  conf_threshold: 0.25
  iou_threshold: 0.45
```

### OCR模型配置
```yaml
ocr_config:
  model_dir: "/path/to/ocr/model"
  intra_op_num_threads: 4
  use_mkldnn: true
  rec_batch_num: 6
  rec_img_h: 32
  rec_img_w: 320
```

### ROI区域配置
```yaml
signalLightROI:  # 信号灯检测区域
  x: 0
  y: 0
  width: 0
  height: 0

timerROI:        # 计时器检测区域
  x: 0
  y: 0
  width: 0
  height: 0
```

### 视频输出配置
```yaml
video_output:
  enable: true
  path: "./output.avi"
  codec: "XVID"
  fps: 30
```

## 运行说明
```bash
./traffic_light_detection [--cfg /path/to/config.yaml]
```

## 状态说明
系统定义了三种运行状态：
- Normal: 系统正常运行，显示当前信号灯颜色和倒计时
- SignalMissing: 未检测到信号灯，显示警告信息
- TimerMissing: 未检测到计时数字，显示警告信息

## 异常处理
- 信号灯检测异常：连续30帧未检测到信号灯时报警
- OCR识别异常：连续60帧未识别到数字时报警

## 可视化输出
- 信号灯检测框（绿色/红色/黄色）
- 当前系统状态（左上角）
- 倒计时数字显示（计时区域下方）
- 异常状态警告

## 注意事项
1. 确保模型文件路径正确配置
2. 检查视频源路径是否正确
3. 根据实际情况调整ROI区域
4. 确保系统有足够的计算资源

## 调试说明
- 程序会在控制台输出详细的运行日志
- 检测结果会保存在 results/output_frames/ 目录下
- 可通过配置文件开启/关闭视频保存功能

## 许可证
Apache License 2.0 