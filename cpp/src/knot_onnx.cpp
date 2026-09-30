// 创建时间：2026-09-28；功能：动态加载口罩打结推理 DLL 并批量检测；目的：验证独立接口和保存逐图结果。

#define NOMINMAX
#include <Windows.h>
#include <opencv2/opencv.hpp>

#include "vt_detect_api.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

double nowMs() {
    LARGE_INTEGER counter = {}, frequency = {};
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * static_cast<double>(counter.QuadPart) /
           static_cast<double>(frequency.QuadPart);
}

fs::path executableDirectory() {
    std::array<wchar_t, 32768> buffer = {};
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                            static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error("Cannot resolve executable directory");
    }
    return fs::path(buffer.data()).parent_path();
}

fs::path repositoryDirectory() {
    fs::path directory = executableDirectory();
    while (!directory.empty()) {
        if (fs::is_regular_file(directory / L"cpp" / L"YOLO26.sln")) return directory;
        const fs::path parent = directory.parent_path();
        if (parent == directory) break;
        directory = parent;
    }
    throw std::runtime_error("Cannot locate repository from executable directory");
}

fs::path defaultModel(const fs::path& repository) {
    for (const auto& relative : {
             fs::path(L"models/v6_knot.onnx"),
             fs::path(L"x64/output/20260929_085458/deploy/v6_knot.onnx"),
             fs::path(L"x64/output/20260928_170714/models/v6_knot.onnx")}) {
        const fs::path candidate = repository / relative;
        if (fs::is_regular_file(candidate)) return candidate;
    }
    throw std::runtime_error("Default model is missing; pass model.onnx explicitly");
}

fs::path defaultImages(const fs::path& repository) {
    for (const auto& relative : {
             fs::path(L"delivery/vt_detect/sample/input"),
             fs::path(L"x64/output/20260929_081042/中文路径/输入"),
             fs::path(L"x64/output/20260928_113627/expanded/dataset/images/val")}) {
        const fs::path candidate = repository / relative;
        if (fs::is_directory(candidate)) return candidate;
    }
    throw std::runtime_error("Default images are missing; pass image_dir explicitly");
}

fs::path localTimestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local = {};
    wchar_t buffer[32] = {};
    if (localtime_s(&local, &now) != 0 ||
        std::wcsftime(buffer, std::size(buffer), L"%Y%m%d_%H%M%S", &local) == 0) {
        throw std::runtime_error("Cannot create local timestamp");
    }
    return fs::path(buffer);
}

struct Api {
    using AbiVersionFn = uint32_t(VT_DETECT_CALL*)();
    using VersionFn = const char*(VT_DETECT_CALL*)();
    using CreateFn = VTDetectHandle(VT_DETECT_CALL*)(const wchar_t*);
    using CreateExFn = VTDetectHandle(VT_DETECT_CALL*)(const wchar_t*,
                                                       const VTDetectOptions*);
    using DetectFn = const VTDetectResult*(VT_DETECT_CALL*)(VTDetectHandle,
                                                             const cv::Mat*);
    using DestroyFn = void(VT_DETECT_CALL*)(VTDetectHandle);
    using LastErrorFn = const char*(VT_DETECT_CALL*)();

    explicit Api(const fs::path& dllPath) {
        module = LoadLibraryExW(dllPath.c_str(), nullptr,
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module) throw std::runtime_error("Cannot load vt_detect.dll and its dependencies");
        try {
            abiVersion = symbol<AbiVersionFn>("VTDetect_GetAbiVersion");
            version = symbol<VersionFn>("VTDetect_GetVersion");
            create = symbol<CreateFn>("VTDetect_Create");
            createEx = symbol<CreateExFn>("VTDetect_CreateEx");
            detect = symbol<DetectFn>("VTDetect_Detect");
            destroy = symbol<DestroyFn>("VTDetect_Destroy");
            lastError = symbol<LastErrorFn>("VTDetect_LastError");
            if (abiVersion() != VT_DETECT_ABI_VERSION) {
                throw std::runtime_error("vt_detect ABI version mismatch");
            }
        } catch (...) {
            FreeLibrary(module);
            module = nullptr;
            throw;
        }
    }
    ~Api() { if (module) FreeLibrary(module); }
    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;

    template <typename T> T symbol(const char* name) const {
        auto address = GetProcAddress(module, name);
        if (!address) throw std::runtime_error(std::string("Missing DLL export: ") + name);
        return reinterpret_cast<T>(address);
    }

    HMODULE module = nullptr;
    AbiVersionFn abiVersion = nullptr;
    VersionFn version = nullptr;
    CreateFn create = nullptr;
    CreateExFn createEx = nullptr;
    DetectFn detect = nullptr;
    DestroyFn destroy = nullptr;
    LastErrorFn lastError = nullptr;
};

struct DetectorHandle {
    Api& api;
    VTDetectHandle value = nullptr;
    explicit DetectorHandle(Api& source) : api(source) {}
    ~DetectorHandle() { if (value) api.destroy(value); }
    DetectorHandle(const DetectorHandle&) = delete;
    DetectorHandle& operator=(const DetectorHandle&) = delete;
};

cv::Mat readImage(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) {
        throw std::runtime_error("Cannot read image: " + path.u8string());
    }
    const auto size = static_cast<size_t>(input.tellg());
    input.seekg(0);
    std::vector<uchar> bytes(size);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!input) throw std::runtime_error("Cannot read complete image");
    const cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Cannot decode image");
    return image;
}

cv::Mat writeImage(const cv::Mat& image, const VTDetectBox* boxes, int count,
                   const fs::path& target) {
    cv::Mat output = image.clone();
    for (int index = 0; index < count; ++index) {
        const auto& box = boxes[index];
        const cv::Point first(static_cast<int>(std::round(box.x1)),
                              static_cast<int>(std::round(box.y1)));
        const cv::Point second(static_cast<int>(std::round(box.x2)),
                               static_cast<int>(std::round(box.y2)));
        cv::rectangle(output, first, second, cv::Scalar(0, 0, 255), 2);
        cv::putText(output, cv::format("NG %.3f", box.confidence),
                    cv::Point(first.x, std::max(20, first.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
    if (count == 0) {
        cv::putText(output, "No detection", cv::Point(16, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 180, 0), 2);
    }
    std::vector<uchar> encoded;
    if (!cv::imencode(".jpg", output, encoded)) {
        throw std::runtime_error("Cannot encode result image");
    }
    fs::create_directories(target.parent_path());
    std::ofstream stream(target, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
    if (!stream) throw std::runtime_error("Cannot save result image");
    return output;
}

bool isImage(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

std::string csvCell(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '"') escaped += '"';
        escaped += character;
    }
    return escaped + '"';
}

void run(const fs::path& modelPath, const fs::path& imageRoot,
         const fs::path& outputRoot, int threads, bool preview) {
    if (!fs::is_regular_file(modelPath) || !fs::is_directory(imageRoot)) {
        throw std::runtime_error("Model or image directory does not exist");
    }
    std::vector<fs::path> images;
    for (const auto& entry : fs::recursive_directory_iterator(imageRoot)) {
        if (entry.is_regular_file() && isImage(entry.path())) images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) throw std::runtime_error("Image directory is empty");

    Api api(executableDirectory() / L"vt_detect.dll");
    DetectorHandle detector(api);
    if (threads == 0) {
        detector.value = api.create(modelPath.c_str());
    } else {
        VTDetectOptions options = {sizeof(VTDetectOptions), threads, 0.25F,
                                   0.5F, 100};
        detector.value = api.createEx(modelPath.c_str(), &options);
    }
    if (!detector.value) {
        throw std::runtime_error(std::string("VTDetect_Create: ") + api.lastError());
    }
    printf("vt_detect_version=%s abi=%u\n", api.version(), api.abiVersion());

    // 第一张真实图用于预热；检测结果由 DLL 内部持有。
    const cv::Mat warmImage = readImage(images.front());
    for (int index = 0; index < 10; ++index) {
        if (!api.detect(detector.value, &warmImage)) {
            throw std::runtime_error(std::string("VTDetect_Detect warmup: ") +
                                     api.lastError());
        }
    }

    fs::create_directories(outputRoot);
    std::ofstream csv(outputRoot / "predictions.csv", std::ios::binary);
    if (!csv) throw std::runtime_error("Cannot create predictions.csv");
    csv << "split,file,box_index,x1,y1,x2,y2,confidence,read_ms,preprocess_ms,inference_ms,postprocess_ms\n";
    csv.setf(std::ios::fixed);
    csv.precision(6);
    std::vector<double> inferenceTimes;
    inferenceTimes.reserve(images.size());
    size_t imageCount = 0;
    for (const auto& path : images) {
        const double started = nowMs();
        const cv::Mat image = readImage(path);
        const double readAt = nowMs();
        const VTDetectResult* result = api.detect(detector.value, &image);
        if (!result) {
            throw std::runtime_error(std::string("VTDetect_Detect: ") + api.lastError());
        }
        const int32_t boxCount = result->box_count;
        const VTDetectBox* boxes = result->boxes;
        inferenceTimes.push_back(result->inference_ms);
        printf("infer time: %.4f ms\n", result->inference_ms);
        const std::string split = csvCell(path.parent_path().filename().u8string());
        const std::string file = csvCell(path.filename().u8string());
        const double readMs = readAt - started;
        if (boxCount == 0) {
            csv << split << ',' << file << ",,,,,,," << readMs << ','
                << result->preprocess_ms << ',' << result->inference_ms << ','
                << result->postprocess_ms << '\n';
        } else {
            for (int32_t index = 0; index < boxCount; ++index) {
                const auto& box = boxes[index];
                csv << split << ',' << file << ',' << index << ',' << box.x1 << ',' << box.y1
                    << ',' << box.x2 << ',' << box.y2 << ',' << box.confidence << ',' << readMs
                    << ',' << result->preprocess_ms << ',' << result->inference_ms << ','
                    << result->postprocess_ms << '\n';
            }
        }
        const fs::path overlayPath = outputRoot / "overlays" /
            path.parent_path().filename() / (path.stem().wstring() + L".jpg");
        const cv::Mat overlay = writeImage(image, boxes, boxCount, overlayPath);
        if (preview) {
            printf("overlay=%s\n", overlayPath.u8string().c_str());
            if (IsDebuggerPresent()) {
                // 仅无参 F5 调试时等待按键，避免批量命令行被预览窗口阻塞。
                try {
                    std::string str_text = "infertime:"+ std::to_string(result->preprocess_ms) + "|" + std::to_string(result->inference_ms) + "|" + std::to_string(result->postprocess_ms);
                    cv::putText(overlay, str_text, { 20,200 }, cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 1, { 128,255,0 });
                    cv::imshow("vt_detect preview - press any key", overlay);

                    cv::waitKey(0);
                    cv::destroyWindow("vt_detect preview - press any key");
                } catch (const cv::Exception& error) {
                    fprintf(stderr, "Preview unavailable: %s\n", error.what());
                }
            }
        }
        ++imageCount;
        if (imageCount % 20 == 0 || imageCount == images.size()) {
            printf("[进度] %zu/%zu\n", imageCount, images.size());
        }
    }
    std::sort(inferenceTimes.begin(), inferenceTimes.end());
    const double mean = std::accumulate(inferenceTimes.begin(), inferenceTimes.end(), 0.0) /
                        inferenceTimes.size();
    printf("images=%zu onnx_cpu_mean_ms=%.4f onnx_cpu_median_ms=%.4f\n",
           imageCount, mean, inferenceTimes[inferenceTimes.size() / 2]);
    // 显式展示销毁调用；异常路径仍由 DetectorHandle 兜底释放。
    api.destroy(detector.value);
    detector.value = nullptr;
    printf("VTDetect_Destroy done\n");
    printf("result_dir=%s\n", outputRoot.u8string().c_str());
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    SetConsoleOutputCP(CP_UTF8);
    try {
        if (argc == 1) {
            // 无参模式用于 VS2019 F5：使用已验证的单图和默认建模参数。
            const fs::path repository = repositoryDirectory();
            const fs::path output = repository / L"x64" / L"output" /
                localTimestamp() / L"vs_debug";
            run(defaultModel(repository), defaultImages(repository), output, 0, true);
            return 0;
        }
        if (argc != 4 && argc != 5) {
            fprintf(stderr, "Usage: KnotOnnxDetect model.onnx image_dir output_dir [cpu_threads]\n");
            return 2;
        }
        const int threads = argc == 5 ? std::stoi(argv[4]) : 0;
        if (threads < 0) throw std::runtime_error("CPU threads must be positive when specified");
        run(argv[1], argv[2], argv[3], threads, false);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
