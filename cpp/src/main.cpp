// 创建时间：2026-09-23；功能：直接执行 ONNX Runtime CPU 或 TensorRT FP32 推理；目的：比较 C++ 与 Python 运行时调用开销。

#define NOMINMAX
#include <Windows.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kInputCount = 3ULL * 640 * 640;

std::vector<float> loadInput(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != kInputCount * sizeof(float)) {
        throw std::runtime_error("输入文件不存在或尺寸不是 1x3x640x640 FP32");
    }
    input.seekg(0);
    std::vector<float> pixels(kInputCount);
    input.read(reinterpret_cast<char*>(pixels.data()),
               static_cast<std::streamsize>(pixels.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("读取输入文件失败");
    }
    return pixels;
}

std::vector<char> loadEngine(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) {
        throw std::runtime_error("TensorRT engine 不存在或为空");
    }
    const auto size = static_cast<size_t>(input.tellg());
    input.seekg(0);
    std::vector<char> bytes(size);
    input.read(bytes.data(), static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("读取 TensorRT engine 失败");
    }
    if (bytes.size() > 5) {
        const uint32_t headerSize = static_cast<uint8_t>(bytes[0])
            | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[1])) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[2])) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[3])) << 24);
        if (headerSize > 0 && headerSize < bytes.size() - 4 && bytes[4] == '{') {
            bytes.erase(bytes.begin(), bytes.begin() + 4 + headerSize);
        }
    }
    return bytes;
}

double tickMs(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER frequency) {
    return 1000.0 * static_cast<double>(end.QuadPart - start.QuadPart)
           / static_cast<double>(frequency.QuadPart);
}

struct Timing {
    double meanMs = 0.0;
    double medianMs = 0.0;
    double p95Ms = 0.0;
};

Timing summarize(std::vector<double> samples) {
    if (samples.empty()) {
        throw std::runtime_error("没有计时样本");
    }
    Timing result;
    result.meanMs = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::sort(samples.begin(), samples.end());
    result.medianMs = samples[samples.size() / 2];
    result.p95Ms = samples[std::min(samples.size() - 1,
                                    static_cast<size_t>(samples.size() * 0.95))];
    return result;
}

void printResult(const char* backend, const Timing& timing, double outputSum) {
    std::cout << std::fixed << std::setprecision(3)
              << "backend=" << backend << " mean_ms=" << timing.meanMs
              << " median_ms=" << timing.medianMs << " p95_ms=" << timing.p95Ms
              << " output_sum=" << outputSum << '\n';
}

void runOnnx(const std::string& modelPath, const std::vector<float>& input,
             int nThreads, int nWarmup, int nIterations) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_cpp_bench");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(nThreads);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const std::wstring widePath(modelPath.begin(), modelPath.end());
    Ort::Session session(env, widePath.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto inputName = session.GetInputNameAllocated(0, allocator);
    const auto outputName = session.GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};
    const int64_t shape[] = {1, 3, 640, 640};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()),
                                                   input.size(), shape, 4);
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    std::vector<double> samples;
    samples.reserve(nIterations);
    double outputSum = 0.0;
    for (int n = 0; n < nWarmup + nIterations; ++n) {
        LARGE_INTEGER start, end;
        QueryPerformanceCounter(&start);
        auto output = session.Run(Ort::RunOptions{nullptr}, inputNames, &tensor, 1,
                                  outputNames, 1);
        QueryPerformanceCounter(&end);
        if (n >= nWarmup) {
            samples.push_back(tickMs(start, end, frequency));
        }
        if (n == nWarmup) {
            const auto count = output[0].GetTensorTypeAndShapeInfo().GetElementCount();
            const float* values = output[0].GetTensorData<float>();
            outputSum = std::accumulate(values, values + count, 0.0);
        }
    }
    printResult("onnx_cpu", summarize(std::move(samples)), outputSum);
}

std::wstring executableDirectory() {
    wchar_t path[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, path, MAX_PATH) == 0) {
        throw std::runtime_error("无法读取程序目录");
    }
    std::wstring result(path);
    return result.substr(0, result.find_last_of(L"\\/"));
}

struct DynamicLibrary {
    HMODULE handle = nullptr;
    explicit DynamicLibrary(const std::wstring& path) : handle(LoadLibraryW(path.c_str())) {
        if (!handle) {
            throw std::runtime_error("无法加载 TensorRT/CUDA DLL，请核对运行时目录和 PATH");
        }
    }
    ~DynamicLibrary() { FreeLibrary(handle); }
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    template <typename T> T symbol(const char* name) const {
        auto ptr = reinterpret_cast<T>(GetProcAddress(handle, name));
        if (!ptr) {
            throw std::runtime_error(std::string("DLL 缺少函数：") + name);
        }
        return ptr;
    }
};

struct CudaApi {
    DynamicLibrary library;
    decltype(&cudaSetDevice) setDevice;
    decltype(&cudaMalloc) mallocDevice;
    decltype(&cudaFree) freeDevice;
    decltype(&cudaMemcpyAsync) memcpyAsync;
    decltype(&cudaStreamCreate) streamCreate;
    decltype(&cudaStreamSynchronize) streamSynchronize;
    decltype(&cudaStreamDestroy) streamDestroy;

    explicit CudaApi(const std::wstring& directory)
        : library(directory + L"\\cudart64_12.dll"),
          setDevice(library.symbol<decltype(setDevice)>("cudaSetDevice")),
          mallocDevice(library.symbol<decltype(mallocDevice)>("cudaMalloc")),
          freeDevice(library.symbol<decltype(freeDevice)>("cudaFree")),
          memcpyAsync(library.symbol<decltype(memcpyAsync)>("cudaMemcpyAsync")),
          streamCreate(library.symbol<decltype(streamCreate)>("cudaStreamCreate")),
          streamSynchronize(library.symbol<decltype(streamSynchronize)>("cudaStreamSynchronize")),
          streamDestroy(library.symbol<decltype(streamDestroy)>("cudaStreamDestroy")) {}

    void check(cudaError_t code, const char* operation) const {
        if (code != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + " 失败，CUDA 错误码 "
                                     + std::to_string(static_cast<int>(code)));
        }
    }
};

struct DeviceMemory {
    CudaApi& cuda;
    void* data = nullptr;
    DeviceMemory(CudaApi& api, size_t bytes) : cuda(api) {
        cuda.check(cuda.mallocDevice(&data, bytes), "cudaMalloc");
    }
    ~DeviceMemory() { if (data) cuda.freeDevice(data); }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
};

struct Stream {
    CudaApi& cuda;
    cudaStream_t handle = nullptr;
    explicit Stream(CudaApi& api) : cuda(api) {
        cuda.check(cuda.streamCreate(&handle), "cudaStreamCreate");
    }
    ~Stream() { if (handle) cuda.streamDestroy(handle); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

size_t tensorCount(nvinfer1::Dims shape) {
    size_t count = 1;
    for (int n = 0; n < shape.nbDims; ++n) {
        if (shape.d[n] <= 0) {
            throw std::runtime_error("TensorRT engine 含动态尺寸，本测试仅支持静态 batch=1");
        }
        count *= static_cast<size_t>(shape.d[n]);
    }
    return count;
}

void runTensorRt(const std::string& modelPath, const std::vector<float>& input,
                 int nWarmup, int nIterations) {
    const std::wstring directory = executableDirectory();
    CudaApi cuda(directory);
    cuda.check(cuda.setDevice(0), "cudaSetDevice");
    DynamicLibrary trt(directory + L"\\nvinfer_10.dll");
    using Factory = void* (__cdecl*)(void*, int32_t) noexcept;
    const auto factory = trt.symbol<Factory>("createInferRuntime_INTERNAL");
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        static_cast<nvinfer1::IRuntime*>(factory(&logger, NV_TENSORRT_VERSION)));
    if (!runtime) {
        throw std::runtime_error("创建 TensorRT 10.16 Runtime 失败");
    }
    const auto bytes = loadEngine(modelPath);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine || engine->getNbIOTensors() != 2) {
        throw std::runtime_error("TensorRT engine 反序列化失败或输入输出数量不符");
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("创建 TensorRT 执行上下文失败");
    }
    const char* inputName = nullptr;
    const char* outputName = nullptr;
    for (int n = 0; n < engine->getNbIOTensors(); ++n) {
        const char* name = engine->getIOTensorName(n);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
        } else {
            outputName = name;
        }
        if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("TensorRT 输入输出不是 FP32");
        }
    }
    if (!inputName || !outputName || tensorCount(engine->getTensorShape(inputName)) != input.size()) {
        throw std::runtime_error("TensorRT 输入尺寸与数据不符");
    }
    const size_t outputCount = tensorCount(engine->getTensorShape(outputName));
    DeviceMemory deviceInput(cuda, input.size() * sizeof(float));
    DeviceMemory deviceOutput(cuda, outputCount * sizeof(float));
    Stream stream(cuda);
    if (!context->setTensorAddress(inputName, deviceInput.data)
        || !context->setTensorAddress(outputName, deviceOutput.data)) {
        throw std::runtime_error("设置 TensorRT 输入输出地址失败");
    }
    cuda.check(cuda.memcpyAsync(deviceInput.data, input.data(), input.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream.handle), "复制输入到 GPU");
    cuda.check(cuda.streamSynchronize(stream.handle), "输入同步");
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    std::vector<double> samples;
    samples.reserve(nIterations);
    for (int n = 0; n < nWarmup + nIterations; ++n) {
        LARGE_INTEGER start, end;
        QueryPerformanceCounter(&start);
        if (!context->enqueueV3(stream.handle)) {
            throw std::runtime_error("TensorRT enqueueV3 失败");
        }
        cuda.check(cuda.streamSynchronize(stream.handle), "推理同步");
        QueryPerformanceCounter(&end);
        if (n >= nWarmup) {
            samples.push_back(tickMs(start, end, frequency));
        }
    }
    std::vector<float> output(outputCount);
    cuda.check(cuda.memcpyAsync(output.data(), deviceOutput.data,
                                output.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                stream.handle), "复制输出到 CPU");
    cuda.check(cuda.streamSynchronize(stream.handle), "输出同步");
    const double outputSum = std::accumulate(output.begin(), output.end(), 0.0);
    printResult("tensorrt", summarize(std::move(samples)), outputSum);
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            std::cerr << "用法：YoloCppBench onnx|trt 模型路径 输入bin 线程数 预热次数 测量次数\n";
            return 2;
        }
        const std::string backend = argv[1];
        const int nThreads = std::stoi(argv[4]);
        const int nWarmup = std::stoi(argv[5]);
        const int nIterations = std::stoi(argv[6]);
        if (nThreads <= 0 || nWarmup < 0 || nIterations <= 0) {
            throw std::runtime_error("线程数、预热次数或测量次数无效");
        }
        const auto input = loadInput(argv[3]);
        if (backend == "onnx") {
            runOnnx(argv[2], input, nThreads, nWarmup, nIterations);
        } else if (backend == "trt") {
            runTensorRt(argv[2], input, nWarmup, nIterations);
        } else {
            throw std::runtime_error("未知推理后端");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
}
