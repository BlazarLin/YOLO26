// 创建时间：2026-09-29；功能：封装检测模型的加载与推理；目的：为动态加载程序提供统一接口。

#define NOMINMAX
#include <Windows.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "vt_detect_api.h"
#include "vt_detect_version.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr int kStride = 32;
thread_local std::string g_lastError;

double nowMs() {
    LARGE_INTEGER counter = {}, frequency = {};
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * static_cast<double>(counter.QuadPart) /
           static_cast<double>(frequency.QuadPart);
}

struct Prepared {
    int width = 0;
    int height = 0;
    int left = 0;
    int top = 0;
    double ratio = 1.0;
};

struct Candidate {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float confidence = 0.0F;
};

Prepared prepare(const cv::Mat& image, std::vector<float>& pixels) {
    Prepared result;
    result.ratio = std::min(static_cast<double>(kInputSize) / image.rows,
                            static_cast<double>(kInputSize) / image.cols);
    const int resizedWidth = static_cast<int>(std::round(image.cols * result.ratio));
    const int resizedHeight = static_cast<int>(std::round(image.rows * result.ratio));
    const int paddingWidth = (kInputSize - resizedWidth) % kStride;
    const int paddingHeight = (kInputSize - resizedHeight) % kStride;
    result.left = static_cast<int>(std::round(paddingWidth / 2.0 - 0.1));
    result.top = static_cast<int>(std::round(paddingHeight / 2.0 - 0.1));
    const int right = static_cast<int>(std::round(paddingWidth / 2.0 + 0.1));
    const int bottom = static_cast<int>(std::round(paddingHeight / 2.0 + 0.1));
    cv::Mat resized, padded, rgb;
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, padded, result.top, bottom, result.left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    result.width = rgb.cols;
    result.height = rgb.rows;
    const size_t plane = static_cast<size_t>(result.width) * result.height;
    pixels.resize(3 * plane);
    for (int y = 0; y < result.height; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < result.width; ++x) {
            const size_t index = static_cast<size_t>(y) * result.width + x;
            for (int channel = 0; channel < 3; ++channel) {
                pixels[static_cast<size_t>(channel) * plane + index] =
                    row[x][channel] / 255.0F;
            }
        }
    }
    return result;
}

float boxIou(const Candidate& a, const Candidate& b) {
    const float width = std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float height = std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float areaA = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float areaB = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    return width * height / std::max(areaA + areaB - width * height, 1e-9F);
}

std::vector<VTDetectBox> decode(const float* values, const std::vector<int64_t>& shape,
                                const Prepared& input, int imageWidth, int imageHeight,
                                const VTDetectOptions& options) {
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 5 || shape[2] <= 0) {
        throw std::runtime_error("ONNX output must be [1,5,anchors]");
    }
    const size_t anchors = static_cast<size_t>(shape[2]);
    std::vector<Candidate> candidates;
    candidates.reserve(256);
    for (size_t index = 0; index < anchors; ++index) {
        const float confidence = values[4 * anchors + index];
        if (confidence < options.confidence) continue;
        const float halfWidth = values[2 * anchors + index] * 0.5F;
        const float halfHeight = values[3 * anchors + index] * 0.5F;
        const float x = values[index];
        const float y = values[anchors + index];
        candidates.push_back({x - halfWidth, y - halfHeight,
                              x + halfWidth, y + halfHeight, confidence});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                         return a.confidence > b.confidence;
                     });
    std::vector<Candidate> selected;
    selected.reserve(std::min(candidates.size(),
                              static_cast<size_t>(options.max_detections)));
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& kept : selected) {
            if (boxIou(candidate, kept) > options.nms_iou) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        selected.push_back(candidate);
        if (selected.size() == static_cast<size_t>(options.max_detections)) break;
    }
    std::vector<VTDetectBox> boxes;
    boxes.reserve(selected.size());
    for (const auto& box : selected) {
        boxes.push_back({
            std::clamp(static_cast<float>((box.x1 - input.left) / input.ratio),
                       0.0F, static_cast<float>(imageWidth)),
            std::clamp(static_cast<float>((box.y1 - input.top) / input.ratio),
                       0.0F, static_cast<float>(imageHeight)),
            std::clamp(static_cast<float>((box.x2 - input.left) / input.ratio),
                       0.0F, static_cast<float>(imageWidth)),
            std::clamp(static_cast<float>((box.y2 - input.top) / input.ratio),
                       0.0F, static_cast<float>(imageHeight)),
            box.confidence, 0});
    }
    return boxes;
}

struct Detector {
    explicit Detector(const wchar_t* modelPath, const VTDetectOptions& config)
        : optionsConfig(config), env(ORT_LOGGING_LEVEL_WARNING, "vt_detect"),
          memory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        options.SetIntraOpNumThreads(config.cpu_threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, modelPath, options);
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
            throw std::runtime_error("ONNX must have one input and one output");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        inputName = session->GetInputNameAllocated(0, allocator).get();
        outputName = session->GetOutputNameAllocated(0, allocator).get();
        const auto inputInfo = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto outputInfo = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto inputShape = inputInfo.GetShape();
        const auto outputShape = outputInfo.GetShape();
        if (inputShape.size() != 4 || inputShape[1] != 3 ||
            inputShape[2] > 0 || inputShape[3] > 0 ||
            outputShape.size() != 3 || outputShape[1] != 5 ||
            inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Expected dynamic FP32 input and one-class raw output");
        }
    }

    VTDetectOptions optionsConfig = {};
    Ort::Env env;
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory;
    std::string inputName;
    std::string outputName;
    std::vector<float> pixels;
    std::vector<VTDetectBox> detections;
    VTDetectResult view = {};
};

void setError(const char* message) {
    g_lastError = message ? message : "unknown error";
}

}  // namespace

extern "C" uint32_t VT_DETECT_CALL VTDetect_GetAbiVersion(void) {
    return VT_DETECT_ABI_VERSION;
}

extern "C" const char* VT_DETECT_CALL VTDetect_GetVersion(void) {
    return VT_DETECT_VERSION_TEXT;
}

extern "C" VTDetectHandle VT_DETECT_CALL VTDetect_Create(const wchar_t* model_path) {
    return VTDetect_CreateEx(model_path, nullptr);
}

extern "C" VTDetectHandle VT_DETECT_CALL VTDetect_CreateEx(
    const wchar_t* model_path, const VTDetectOptions* options) {
    g_lastError.clear();
    const VTDetectOptions defaults = {sizeof(VTDetectOptions), 4, 0.25F, 0.5F, 100};
    const VTDetectOptions& config = options ? *options : defaults;
    if (!model_path || !*model_path || config.struct_size != sizeof(VTDetectOptions) ||
        config.cpu_threads <= 0 || !std::isfinite(config.confidence) ||
        !std::isfinite(config.nms_iou) || config.confidence < 0.0F ||
        config.confidence > 1.0F || config.nms_iou <= 0.0F ||
        config.nms_iou >= 1.0F || config.max_detections <= 0 ||
        config.max_detections > 1000) {
        setError("Invalid model path or options");
        return nullptr;
    }
    try {
        return new Detector(model_path, config);
    } catch (const std::exception& error) {
        setError(error.what());
        return nullptr;
    } catch (...) {
        setError("Unknown model load failure");
        return nullptr;
    }
}

extern "C" const VTDetectResult* VT_DETECT_CALL VTDetect_Detect(
    VTDetectHandle handle, const cv::Mat* image) {
    g_lastError.clear();
    if (!handle || !image || image->empty() || image->dims != 2 ||
        image->type() != CV_8UC3) {
        setError("Expected nonempty CV_8UC3 BGR image and valid handle");
        return nullptr;
    }
    auto& detector = *static_cast<Detector*>(handle);
    detector.detections.clear();
    detector.view = {};
    try {
        const double started = nowMs();
        const Prepared prepared = prepare(*image, detector.pixels);
        const double preparedAt = nowMs();
        const std::array<int64_t, 4> shape = {1, 3, prepared.height, prepared.width};
        auto tensor = Ort::Value::CreateTensor<float>(
            detector.memory, detector.pixels.data(), detector.pixels.size(),
            shape.data(), shape.size());
        const char* inputNames[] = {detector.inputName.c_str()};
        const char* outputNames[] = {detector.outputName.c_str()};
        auto result = detector.session->Run(Ort::RunOptions{nullptr}, inputNames,
                                            &tensor, 1, outputNames, 1);
        const double inferredAt = nowMs();
        detector.detections = decode(result[0].GetTensorData<float>(),
                                     result[0].GetTensorTypeAndShapeInfo().GetShape(),
                                     prepared, image->cols, image->rows,
                                     detector.optionsConfig);
        const double decodedAt = nowMs();
        detector.view.boxes = detector.detections.empty() ? nullptr :
                              detector.detections.data();
        detector.view.box_count = static_cast<int32_t>(detector.detections.size());
        detector.view.preprocess_ms = preparedAt - started;
        detector.view.inference_ms = inferredAt - preparedAt;
        detector.view.postprocess_ms = decodedAt - inferredAt;
        return &detector.view;
    } catch (const std::exception& error) {
        setError(error.what());
        return nullptr;
    } catch (...) {
        setError("Unknown inference failure");
        return nullptr;
    }
}

extern "C" void VT_DETECT_CALL VTDetect_Destroy(VTDetectHandle handle) {
    delete static_cast<Detector*>(handle);
}

extern "C" const char* VT_DETECT_CALL VTDetect_LastError(void) {
    return g_lastError.c_str();
}
