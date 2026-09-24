// 创建时间：2026-09-23；功能：执行 YOLOv8-n 与 YOLO26-n 的 C++ 完整检测；目的：测量含前后处理的 CPU ONNX 和 TensorRT 耗时。

#define NOMINMAX
#include <Windows.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSize = 640;
constexpr float kConfidence = 0.25F;
constexpr float kIou = 0.7F;

struct Prepared {
    std::vector<float> pixels;
    float ratio = 1.0F;
    int left = 0;
    int top = 0;
};

Prepared prepare(const cv::Mat& source, float* destination = nullptr) {
    Prepared result;
    result.ratio = std::min(static_cast<float>(kSize) / source.rows,
                            static_cast<float>(kSize) / source.cols);
    const int width = static_cast<int>(std::round(source.cols * result.ratio));
    const int height = static_cast<int>(std::round(source.rows * result.ratio));
    result.left = static_cast<int>(std::round((kSize - width) / 2.0F - 0.1F));
    result.top = static_cast<int>(std::round((kSize - height) / 2.0F - 0.1F));
    cv::Mat resized, padded;
    cv::resize(source, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, padded, result.top, kSize - height - result.top,
                       result.left, kSize - width - result.left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    std::vector<cv::Mat> channels;
    cv::split(padded, channels);
    if (!destination) {
        result.pixels.resize(3ULL * kSize * kSize);
        destination = result.pixels.data();
    }
    const size_t plane = static_cast<size_t>(kSize) * kSize;
    for (int channel = 0; channel < 3; ++channel) {
        cv::Mat output(kSize, kSize, CV_32F, destination + channel * plane);
        channels[2 - channel].convertTo(output, CV_32F, 1.0 / 255.0);
    }
    return result;
}

struct Detection {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float confidence = 0.0F;
    int classId = 0;
};

float overlap(const Detection& first, const Detection& second) {
    const float width = std::max(0.0F, std::min(first.x2, second.x2) - std::max(first.x1, second.x1));
    const float height = std::max(0.0F, std::min(first.y2, second.y2) - std::max(first.y1, second.y1));
    const float intersection = width * height;
    const float firstArea = (first.x2 - first.x1) * (first.y2 - first.y1);
    const float secondArea = (second.x2 - second.x1) * (second.y2 - second.y1);
    return intersection / std::max(firstArea + secondArea - intersection, 1e-6F);
}

void scale(Detection& box, const Prepared& input, const cv::Mat& image) {
    box.x1 = std::clamp((box.x1 - input.left) / input.ratio, 0.0F,
                        static_cast<float>(image.cols));
    box.x2 = std::clamp((box.x2 - input.left) / input.ratio, 0.0F,
                        static_cast<float>(image.cols));
    box.y1 = std::clamp((box.y1 - input.top) / input.ratio, 0.0F,
                        static_cast<float>(image.rows));
    box.y2 = std::clamp((box.y2 - input.top) / input.ratio, 0.0F,
                        static_cast<float>(image.rows));
}

std::vector<Detection> decode(const float* output, size_t count,
                              const Prepared& input, const cv::Mat& image) {
    std::vector<Detection> candidates;
    if (count == 300ULL * 6) {
        for (size_t index = 0; index < 300; ++index) {
            const float* row = output + index * 6;
            if (row[4] < kConfidence) {
                continue;
            }
            Detection box{row[0], row[1], row[2], row[3], row[4],
                          static_cast<int>(row[5])};
            scale(box, input, image);
            candidates.push_back(box);
        }
        return candidates;
    }
    if (count != 84ULL * 8400) {
        throw std::runtime_error("未知输出形状，需使用 640x640 COCO 检测模型");
    }
    for (size_t index = 0; index < 8400; ++index) {
        int classId = 0;
        float confidence = output[4 * 8400 + index];
        for (int label = 1; label < 80; ++label) {
            const float value = output[(4 + label) * 8400 + index];
            if (value > confidence) {
                confidence = value;
                classId = label;
            }
        }
        if (confidence < kConfidence) {
            continue;
        }
        const float x = output[index];
        const float y = output[8400 + index];
        const float halfWidth = output[2 * 8400 + index] * 0.5F;
        const float halfHeight = output[3 * 8400 + index] * 0.5F;
        candidates.push_back({x - halfWidth, y - halfHeight,
                              x + halfWidth, y + halfHeight, confidence, classId});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& first, const Detection& second) {
                  return first.confidence > second.confidence;
              });
    std::vector<Detection> selected;
    selected.reserve(std::min<size_t>(300, candidates.size()));
    for (auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& kept : selected) {
            if (candidate.classId == kept.classId && overlap(candidate, kept) > kIou) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.push_back(candidate);
            if (selected.size() == 300) {
                break;
            }
        }
    }
    for (auto& box : selected) {
        scale(box, input, image);
    }
    return selected;
}

double checksum(const std::vector<Detection>& boxes) {
    double sum = 0.0;
    for (const auto& box : boxes) {
        sum += box.x1 + box.y1 + box.x2 + box.y2 + box.confidence + box.classId;
    }
    return sum;
}

void saveDetections(const cv::Mat& image, const std::vector<Detection>& boxes,
                    const std::string& outputPath) {
    if (outputPath.empty()) return;
    cv::Mat rendered = image.clone();
    for (const auto& box : boxes) {
        const cv::Point first(static_cast<int>(std::round(box.x1)),
                              static_cast<int>(std::round(box.y1)));
        const cv::Point second(static_cast<int>(std::round(box.x2)),
                               static_cast<int>(std::round(box.y2)));
        cv::rectangle(rendered, first, second, cv::Scalar(0, 0, 255), 2);
        const std::string label = "class " + std::to_string(box.classId) + " "
            + cv::format("%.2f", box.confidence);
        cv::putText(rendered, label, cv::Point(first.x, std::max(20, first.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);
    }
    if (boxes.empty()) {
        cv::putText(rendered, "0 detections", cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0, 0, 255), 2);
    }
    if (!cv::imwrite(outputPath, rendered)) {
        throw std::runtime_error("推理结果图保存失败");
    }
}

double nowMs() {
    LARGE_INTEGER value, frequency;
    QueryPerformanceCounter(&value);
    QueryPerformanceFrequency(&frequency);
    return value.QuadPart * 1000.0 / frequency.QuadPart;
}

void report(const std::vector<double>& samples, size_t count, double sum) {
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::cout << std::fixed << std::setprecision(3)
              << "mean_ms=" << mean << " median_ms=" << ordered[ordered.size() / 2]
              << " p95_ms=" << ordered[std::min(ordered.size() - 1,
                                                  static_cast<size_t>(ordered.size() * 0.95))]
              << " count=" << count << " checksum=" << sum << '\n';
}

void runOnnx(const std::string& modelPath, const cv::Mat& image,
             int threads, int warmup, int iterations, const std::string& outputPath) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_cpp_detect");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const std::wstring widePath(modelPath.begin(), modelPath.end());
    Ort::Session session(env, widePath.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto inputName = session.GetInputNameAllocated(0, allocator);
    const auto outputName = session.GetOutputNameAllocated(0, allocator);
    const char* inputs[] = {inputName.get()};
    const char* outputs[] = {outputName.get()};
    const int64_t shape[] = {1, 3, kSize, kSize};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<double> samples;
    samples.reserve(iterations);
    size_t count = 0;
    double sum = 0.0;
    std::vector<Detection> savedBoxes;
    for (int index = 0; index < warmup + iterations; ++index) {
        const double start = nowMs();
        auto prepared = prepare(image);
        auto tensor = Ort::Value::CreateTensor<float>(memory, prepared.pixels.data(),
                                                       prepared.pixels.size(), shape, 4);
        auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        const auto outputCount = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const auto boxes = decode(result[0].GetTensorData<float>(), outputCount, prepared, image);
        const double end = nowMs();
        if (index >= warmup) {
            samples.push_back(end - start);
        }
        if (index == warmup) {
            count = boxes.size();
            sum = checksum(boxes);
            if (!outputPath.empty()) savedBoxes = boxes;
        }
    }
    report(samples, count, sum);
    saveDetections(image, savedBoxes, outputPath);
}

struct Library {
    HMODULE handle = nullptr;
    explicit Library(const std::wstring& path) : handle(LoadLibraryW(path.c_str())) {
        if (!handle) throw std::runtime_error("无法加载 TensorRT/CUDA DLL");
    }
    ~Library() { FreeLibrary(handle); }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    template <typename T> T symbol(const char* name) const {
        auto result = reinterpret_cast<T>(GetProcAddress(handle, name));
        if (!result) throw std::runtime_error(std::string("缺少 DLL 函数：") + name);
        return result;
    }
};

struct Cuda {
    Library library;
    decltype(&cudaSetDevice) setDevice;
    decltype(&cudaMalloc) allocate;
    decltype(&cudaFree) release;
    decltype(&cudaMemcpyAsync) copy;
    decltype(&cudaStreamCreate) createStream;
    decltype(&cudaStreamSynchronize) synchronize;
    decltype(&cudaStreamDestroy) destroyStream;
    decltype(&cudaMallocHost) allocateHost;
    decltype(&cudaFreeHost) releaseHost;
    explicit Cuda(const std::wstring& directory)
        : library(directory + L"\\cudart64_12.dll"),
          setDevice(library.symbol<decltype(setDevice)>("cudaSetDevice")),
          allocate(library.symbol<decltype(allocate)>("cudaMalloc")),
          release(library.symbol<decltype(release)>("cudaFree")),
          copy(library.symbol<decltype(copy)>("cudaMemcpyAsync")),
          createStream(library.symbol<decltype(createStream)>("cudaStreamCreate")),
          synchronize(library.symbol<decltype(synchronize)>("cudaStreamSynchronize")),
          destroyStream(library.symbol<decltype(destroyStream)>("cudaStreamDestroy")),
          allocateHost(library.symbol<decltype(allocateHost)>("cudaMallocHost")),
          releaseHost(library.symbol<decltype(releaseHost)>("cudaFreeHost")) {}
    void check(cudaError_t code) const {
        if (code != cudaSuccess) throw std::runtime_error("CUDA 错误码 " + std::to_string(code));
    }
};

struct GpuMemory {
    Cuda& cuda;
    void* data = nullptr;
    GpuMemory(Cuda& api, size_t bytes) : cuda(api) { cuda.check(cuda.allocate(&data, bytes)); }
    ~GpuMemory() { if (data) cuda.release(data); }
};

struct HostMemory {
    Cuda& cuda;
    void* data = nullptr;
    HostMemory(Cuda& api, size_t bytes) : cuda(api) { cuda.check(cuda.allocateHost(&data, bytes)); }
    ~HostMemory() { if (data) cuda.releaseHost(data); }
};

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "TensorRT: " << message << '\n';
    }
};

size_t tensorCount(nvinfer1::Dims shape) {
    size_t count = 1;
    for (int index = 0; index < shape.nbDims; ++index) {
        if (shape.d[index] <= 0) throw std::runtime_error("仅支持静态 TensorRT engine");
        count *= static_cast<size_t>(shape.d[index]);
    }
    return count;
}

std::vector<char> readEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 4) throw std::runtime_error("无法读取 engine 文件");
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> bytes(size);
    file.read(bytes.data(), static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("engine 文件读取失败");
    const uint32_t header = static_cast<uint8_t>(bytes[0])
        | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[1])) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[2])) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[3])) << 24);
    if (header > 0 && header < bytes.size() - 4 && bytes[4] == '{') {
        bytes.erase(bytes.begin(), bytes.begin() + 4 + header);
    }
    return bytes;
}

void runTrt(const std::string& modelPath, const cv::Mat& image,
            int warmup, int iterations, const std::string& outputPath) {
    wchar_t path[MAX_PATH] = {};
    if (!GetModuleFileNameW(nullptr, path, MAX_PATH)) throw std::runtime_error("程序路径读取失败");
    std::wstring directory(path);
    directory.resize(directory.find_last_of(L"\\/"));
    Cuda cuda(directory);
    cuda.check(cuda.setDevice(0));
    Library trt(directory + L"\\nvinfer_10.dll");
    using Factory = void* (__cdecl*)(void*, int32_t) noexcept;
    Logger logger;
    auto factory = trt.symbol<Factory>("createInferRuntime_INTERNAL");
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        static_cast<nvinfer1::IRuntime*>(factory(&logger, NV_TENSORRT_VERSION)));
    if (!runtime) throw std::runtime_error("创建 TensorRT Runtime 失败");
    const auto bytes = readEngine(modelPath);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine || engine->getNbIOTensors() != 2) throw std::runtime_error("engine 反序列化失败");
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) throw std::runtime_error("创建 TensorRT 执行上下文失败");
    const char* inputName = nullptr;
    const char* outputName = nullptr;
    for (int index = 0; index < 2; ++index) {
        const char* name = engine->getIOTensorName(index);
        if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("仅支持 FP32 engine");
        }
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) inputName = name;
        else outputName = name;
    }
    if (!inputName || !outputName || tensorCount(engine->getTensorShape(inputName)) != 3ULL * kSize * kSize) {
        throw std::runtime_error("engine 输入尺寸不符");
    }
    const size_t outputCount = tensorCount(engine->getTensorShape(outputName));
    GpuMemory input(cuda, 3ULL * kSize * kSize * sizeof(float));
    GpuMemory output(cuda, outputCount * sizeof(float));
    HostMemory hostInput(cuda, 3ULL * kSize * kSize * sizeof(float));
    HostMemory hostOutput(cuda, outputCount * sizeof(float));
    cudaStream_t stream = nullptr;
    cuda.check(cuda.createStream(&stream));
    if (!context->setTensorAddress(inputName, input.data)
        || !context->setTensorAddress(outputName, output.data)) {
        throw std::runtime_error("engine 地址设置失败");
    }
    std::vector<double> samples;
    samples.reserve(iterations);
    double preprocessingMs = 0.0;
    double executionMs = 0.0;
    double postprocessingMs = 0.0;
    size_t count = 0;
    double sum = 0.0;
    std::vector<Detection> savedBoxes;
    for (int index = 0; index < warmup + iterations; ++index) {
        const double start = nowMs();
        const auto prepared = prepare(image, static_cast<float*>(hostInput.data));
        const double preparedAt = nowMs();
        cuda.check(cuda.copy(input.data, hostInput.data,
                             3ULL * kSize * kSize * sizeof(float), cudaMemcpyHostToDevice, stream));
        if (!context->enqueueV3(stream)) throw std::runtime_error("TensorRT 推理失败");
        cuda.check(cuda.copy(hostOutput.data, output.data,
                             outputCount * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cuda.check(cuda.synchronize(stream));
        const double executedAt = nowMs();
        const auto boxes = decode(static_cast<float*>(hostOutput.data), outputCount, prepared, image);
        const double end = nowMs();
        if (index >= warmup) {
            samples.push_back(end - start);
            preprocessingMs += preparedAt - start;
            executionMs += executedAt - preparedAt;
            postprocessingMs += end - executedAt;
        }
        if (index == warmup) {
            count = boxes.size();
            sum = checksum(boxes);
            if (!outputPath.empty()) savedBoxes = boxes;
        }
    }
    cuda.check(cuda.destroyStream(stream));
    report(samples, count, sum);
    std::cout << std::fixed << std::setprecision(3)
              << "preprocess_ms=" << preprocessingMs / iterations
              << " transfer_and_engine_ms=" << executionMs / iterations
              << " postprocess_ms=" << postprocessingMs / iterations << '\n';
    saveDetections(image, savedBoxes, outputPath);
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 8) {
            std::cerr << "用法：YoloCppDetect onnx|trt 模型路径 图片路径 线程数 预热次数 测量次数 [结果图路径]\n";
            return 2;
        }
        const int threads = std::stoi(argv[4]);
        const int warmup = std::stoi(argv[5]);
        const int iterations = std::stoi(argv[6]);
        if (threads <= 0 || warmup < 0 || iterations <= 0) {
            throw std::runtime_error("计时参数无效");
        }
        const cv::Mat image = cv::imread(argv[3]);
        if (image.empty()) throw std::runtime_error("图片读取失败");
        const std::string outputPath = argc == 8 ? argv[7] : "";
        if (std::string(argv[1]) == "onnx") runOnnx(argv[2], image, threads, warmup, iterations, outputPath);
        else if (std::string(argv[1]) == "trt") runTrt(argv[2], image, warmup, iterations, outputPath);
        else throw std::runtime_error("未知推理后端");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
}
