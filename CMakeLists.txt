cmake_minimum_required(VERSION 3.10)
project(TrafficLightDetection)

set(CMAKE_CXX_STANDARD 14)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 yaml-cpp
find_package(yaml-cpp REQUIRED)

# 手动指定 ONNX Runtime 路径
set(ONNXRuntime_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/deps/onnxruntime-linux-x64-1.11.1/include)
set(ONNXRuntime_LIBRARIES ${CMAKE_SOURCE_DIR}/deps/onnxruntime-linux-x64-1.11.1/lib/libonnxruntime.so)

# 手动指定 PaddlePaddle 路径
set(Paddle_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/deps/paddle_inference/paddle/include)
set(Paddle_LIBRARIES ${CMAKE_SOURCE_DIR}/deps/paddle_inference/paddle/lib/libpaddle_inference.so)

# 设置 Paddle Inference 的第三方库路径
set(Paddle_THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/deps/paddle_inference/third_party/install)
set(Paddle_THIRD_PARTY_LIBS
    ${Paddle_THIRD_PARTY_DIR}/mklml/lib/libiomp5.so
    ${Paddle_THIRD_PARTY_DIR}/mklml/lib/libmklml_intel.so
    ${Paddle_THIRD_PARTY_DIR}/onednn/lib/libdnnl.so.3
    ${Paddle_THIRD_PARTY_DIR}/openvino/intel64/libopenvino.so
    ${Paddle_THIRD_PARTY_DIR}/openvino/intel64/libopenvino_paddle_frontend.so
)

# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRuntime_INCLUDE_DIRS})
include_directories(${Paddle_INCLUDE_DIRS})
include_directories(${Paddle_THIRD_PARTY_DIR}/mklml/include)
include_directories(${Paddle_THIRD_PARTY_DIR}/onednn/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(traffic_light_detection
    src/main.cpp
    src/yolo_wrapper.cpp  
    src/ocr.cpp 
    src/data.cpp
    src/op.cpp
    src/utils.cpp
)

# 链接库
target_link_libraries(traffic_light_detection 
    ${OpenCV_LIBS} 
    ${ONNXRuntime_LIBRARIES} 
    ${Paddle_LIBRARIES}
    ${Paddle_THIRD_PARTY_LIBS}
    yaml-cpp
)
