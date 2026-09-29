# 创建时间：2026-09-23；功能：构建 C++17 打结检测 DLL 与历史测速程序；目的：提供独立动态接口和运行验证。
cmake_minimum_required(VERSION 3.20)
project(YoloCppBench LANGUAGES CXX)
if(WIN32)
    enable_language(RC)
endif()

if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(ORT_ROOT "${REPO_ROOT}/third_party/onnxruntime_1_15_1"
    CACHE PATH "ONNX Runtime SDK 根目录")
set(TRT_ROOT "${REPO_ROOT}/third_party/tensorrt")
set(CUDA_ROOT "${REPO_ROOT}/third_party/cuda")
set(OPENCV_ROOT "${REPO_ROOT}/third_party/opencv")
set(CUDA_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include"
    CACHE PATH "CUDA API 头文件目录")
option(YOLO26_BUILD_LEGACY_TRT_TOOLS "构建历史 TensorRT 测速程序" ON)

foreach(required_file
        "${ORT_ROOT}/include/onnxruntime_cxx_api.h"
        "${ORT_ROOT}/lib/onnxruntime.lib"
        "${ORT_ROOT}/dll/onnxruntime.dll"
        "${ORT_ROOT}/dll/onnxruntime_providers_shared.dll")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "缺少依赖：${required_file}")
    endif()
endforeach()

if(YOLO26_BUILD_LEGACY_TRT_TOOLS)
foreach(required_file
        "${TRT_ROOT}/include/NvInferRuntime.h"
        "${TRT_ROOT}/dll/nvinfer_10.dll"
        "${CUDA_ROOT}/dll/cudart64_12.dll"
        "${CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "缺少依赖：${required_file}")
    endif()
endforeach()
endif()

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

if(YOLO26_BUILD_LEGACY_TRT_TOOLS)
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
endif()

add_library(vt_detect SHARED src/vt_detect.cpp src/vt_detect_version.rc)
target_include_directories(vt_detect PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include" "${ORT_ROOT}/include" "${OPENCV_ROOT}/include")
target_link_libraries(vt_detect PRIVATE
    "${ORT_ROOT}/lib/onnxruntime.lib"
    "$<IF:$<CONFIG:Debug>,${OPENCV_ROOT}/lib/opencv_world455d.lib,${OPENCV_ROOT}/lib/opencv_world455.lib>")
target_compile_definitions(vt_detect PRIVATE UNICODE _UNICODE VT_DETECT_BUILD)
if(MSVC)
    set_property(TARGET vt_detect PROPERTY MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
    target_compile_options(vt_detect PRIVATE /utf-8 /FS /EHsc)
    target_link_options(vt_detect PRIVATE /DEBUG)
endif()
foreach(runtime_dll
        "${ORT_ROOT}/dll/onnxruntime.dll"
        "${ORT_ROOT}/dll/onnxruntime_providers_shared.dll")
    add_custom_command(TARGET vt_detect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${runtime_dll}" "$<TARGET_FILE_DIR:vt_detect>")
endforeach()
add_custom_command(TARGET vt_detect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<IF:$<CONFIG:Debug>,${OPENCV_ROOT}/dll/opencv_world455d.dll,${OPENCV_ROOT}/dll/opencv_world455.dll>"
        "$<TARGET_FILE_DIR:vt_detect>")

add_executable(KnotOnnxDetect src/knot_onnx.cpp)
target_include_directories(KnotOnnxDetect PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include" "${OPENCV_ROOT}/include")
target_link_libraries(KnotOnnxDetect PRIVATE
    "$<IF:$<CONFIG:Debug>,${OPENCV_ROOT}/lib/opencv_world455d.lib,${OPENCV_ROOT}/lib/opencv_world455.lib>")
target_compile_definitions(KnotOnnxDetect PRIVATE UNICODE _UNICODE)
add_dependencies(KnotOnnxDetect vt_detect)
if(MSVC)
    set_property(TARGET KnotOnnxDetect PROPERTY MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
    target_compile_options(KnotOnnxDetect PRIVATE /utf-8 /FS /EHsc)
    target_link_options(KnotOnnxDetect PRIVATE /DEBUG)
endif()
add_custom_command(TARGET KnotOnnxDetect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:vt_detect>" "$<TARGET_FILE_DIR:KnotOnnxDetect>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<IF:$<CONFIG:Debug>,${OPENCV_ROOT}/dll/opencv_world455d.dll,${OPENCV_ROOT}/dll/opencv_world455.dll>"
        "$<TARGET_FILE_DIR:KnotOnnxDetect>")
