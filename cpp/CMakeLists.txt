# 创建时间：2026-09-23；功能：构建 C++17 ONNX Runtime 与 TensorRT 推理测速程序；目的：与 Python 直连运行时对照。
cmake_minimum_required(VERSION 3.20)
project(YoloCppBench LANGUAGES CXX)

if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(ORT_ROOT "${REPO_ROOT}/third_party/onnxruntime_1_16_3"
    CACHE PATH "ONNX Runtime SDK 根目录")
set(TRT_ROOT "${REPO_ROOT}/third_party/tensorrt")
set(CUDA_ROOT "${REPO_ROOT}/third_party/cuda")
set(OPENCV_ROOT "${REPO_ROOT}/third_party/opencv")
set(CUDA_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include"
    CACHE PATH "CUDA API 头文件目录")

foreach(required_file
        "${ORT_ROOT}/include/onnxruntime_cxx_api.h"
        "${ORT_ROOT}/lib/onnxruntime.lib"
        "${ORT_ROOT}/dll/onnxruntime.dll"
        "${TRT_ROOT}/include/NvInferRuntime.h"
        "${TRT_ROOT}/dll/nvinfer_10.dll"
        "${CUDA_ROOT}/dll/cudart64_12.dll"
        "${CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "缺少依赖：${required_file}")
    endif()
endforeach()

foreach(required_file
        "${OPENCV_ROOT}/include/opencv2/opencv.hpp"
        "${OPENCV_ROOT}/lib/opencv_world455.lib"
        "${OPENCV_ROOT}/lib/opencv_world455d.lib"
        "${OPENCV_ROOT}/dll/opencv_world455.dll"
        "${OPENCV_ROOT}/dll/opencv_world455d.dll")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "缺少依赖：${required_file}")
    endif()
endforeach()

add_executable(YoloCppBench src/main.cpp)
target_include_directories(YoloCppBench PRIVATE
    "${ORT_ROOT}/include" "${TRT_ROOT}/include" "${CUDA_INCLUDE_DIR}")
target_link_libraries(YoloCppBench PRIVATE "${ORT_ROOT}/lib/onnxruntime.lib")
target_compile_definitions(YoloCppBench PRIVATE UNICODE _UNICODE)

if(MSVC)
    set_property(TARGET YoloCppBench PROPERTY MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
    target_compile_options(YoloCppBench PRIVATE /utf-8 /FS /EHsc)
    target_link_options(YoloCppBench PRIVATE /DEBUG)
endif()

foreach(runtime_dll
        "${ORT_ROOT}/dll/onnxruntime.dll"
        "${ORT_ROOT}/dll/onnxruntime_providers_shared.dll"
        "${TRT_ROOT}/dll/nvinfer_10.dll"
        "${TRT_ROOT}/dll/nvinfer_plugin_10.dll"
        "${CUDA_ROOT}/dll/cudart64_12.dll")
    add_custom_command(TARGET YoloCppBench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime_dll}" "$<TARGET_FILE_DIR:YoloCppBench>")
endforeach()

add_executable(YoloCppDetect src/detect.cpp)
target_include_directories(YoloCppDetect PRIVATE
    "${ORT_ROOT}/include" "${TRT_ROOT}/include" "${CUDA_INCLUDE_DIR}"
    "${OPENCV_ROOT}/include")
target_link_libraries(YoloCppDetect PRIVATE
    "${ORT_ROOT}/lib/onnxruntime.lib"
    "$<IF:$<CONFIG:Debug>,${OPENCV_ROOT}/lib/opencv_world455d.lib,${OPENCV_ROOT}/lib/opencv_world455.lib>")
target_compile_definitions(YoloCppDetect PRIVATE UNICODE _UNICODE)
if(MSVC)
    set_property(TARGET YoloCppDetect PROPERTY MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
    target_compile_options(YoloCppDetect PRIVATE /utf-8 /FS /EHsc)
    target_link_options(YoloCppDetect PRIVATE /DEBUG)
endif()
foreach(runtime_dll
        "${ORT_ROOT}/dll/onnxruntime.dll"
        "${ORT_ROOT}/dll/onnxruntime_providers_shared.dll"
        "${TRT_ROOT}/dll/nvinfer_10.dll"
        "${TRT_ROOT}/dll/nvinfer_plugin_10.dll"
        "${CUDA_ROOT}/dll/cudart64_12.dll")
    add_custom_command(TARGET YoloCppDetect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime_dll}" "$<TARGET_FILE_DIR:YoloCppDetect>")
endforeach()
add_custom_command(TARGET YoloCppDetect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<IF:$<CONFIG:Debug>,${OPENCV_ROOT}/dll/opencv_world455d.dll,${OPENCV_ROOT}/dll/opencv_world455.dll>"
        "$<TARGET_FILE_DIR:YoloCppDetect>")
