// 创建时间：2026-09-29；功能：vt_detect 动态检测接口；目的：以少量参数供 VS2019 程序动态加载模型并检测图像。
#pragma once

#include <stdint.h>
#include <opencv2/core/mat.hpp>

#if defined(_WIN32)
#if defined(VT_DETECT_BUILD)
#define VT_DETECT_API __declspec(dllexport)
#else
#define VT_DETECT_API __declspec(dllimport)
#endif
#define VT_DETECT_CALL __cdecl
#else
#define VT_DETECT_API
#define VT_DETECT_CALL
#endif

enum { VT_DETECT_ABI_VERSION = 1 };

typedef void* VTDetectHandle;

typedef struct VTDetectOptions {
    uint32_t struct_size;
    int32_t cpu_threads;
    float confidence;
    float nms_iou;
    int32_t max_detections;
} VTDetectOptions;

typedef struct VTDetectBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t class_id;
} VTDetectBox;

typedef struct VTDetectResult {
    const VTDetectBox* boxes;
    int32_t box_count;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
} VTDetectResult;

extern "C" {

// 先核对接口版本；软件版本与接口版本分开管理。
VT_DETECT_API uint32_t VT_DETECT_CALL VTDetect_GetAbiVersion(void);
VT_DETECT_API const char* VT_DETECT_CALL VTDetect_GetVersion(void);

// 默认参数：CPU 4 线程、置信度 0.25、NMS IoU 0.5、最多 100 个检测框。
// 模型路径为 UTF-16；失败返回空句柄，并通过 VTDetect_LastError 查询原因。
VT_DETECT_API VTDetectHandle VT_DETECT_CALL VTDetect_Create(const wchar_t* model_path);
VT_DETECT_API VTDetectHandle VT_DETECT_CALL VTDetect_CreateEx(
    const wchar_t* model_path, const VTDetectOptions* options);

// 目前接收 CV_8UC3 BGR 图像；尺寸、行跨度和预处理参数由 DLL 读取。
// 结果由句柄持有，在该句柄下次 Detect 或 Destroy 前有效；每个工作线程单独创建句柄。
VT_DETECT_API const VTDetectResult* VT_DETECT_CALL VTDetect_Detect(
    VTDetectHandle handle, const cv::Mat* image);

VT_DETECT_API void VT_DETECT_CALL VTDetect_Destroy(VTDetectHandle handle);

// 当前线程最近一次失败的 UTF-8 文本；下次 API 调用可能覆盖。
VT_DETECT_API const char* VT_DETECT_CALL VTDetect_LastError(void);

}
