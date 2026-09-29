# 第三方依赖放置说明

本目录仅跟踪此说明，`include/`、`lib/`、`dll/` 和所有二进制文件均不提交。以下版本是本机已验证的组合；下载时优先选择相同版本和 Windows x64 包。

| 目录 | 版本与用途 | 官方获取位置 | 需要放置的文件 |
| --- | --- | --- | --- |
| `onnxruntime_1_15_1/` | ONNX Runtime CPU 1.15.1，当前 C++ 默认版本；源包的 `VERSION_NUMBER` 与 DLL 均已核对 | 本机 `G:\Code\2607-RapidOCR\third_party\RapidOcrOnnx\dependencies\onnxruntime-win-x64-1.15.1`；外部机器从 [Microsoft ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 取相同版本的 Windows x64 CPU 包 | 源包 `include/` 全部头文件，`lib/onnxruntime.lib`，`lib/onnxruntime.dll` 和 `lib/onnxruntime_providers_shared.dll` 后两者放本目录 `dll/` |
| `onnxruntime_1_16_3/` | ONNX Runtime CPU 1.16.3，历史 C++ 实验保留 | [Microsoft 1.16.3 Release](https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3)；本机也可从 `E:\CodingLife\PPOCR_V6\thrid_part\onnxruntime` 复制 | `include/onnxruntime_cxx_api.h` 等头文件、`lib/onnxruntime.lib`、`dll/onnxruntime.dll`、`dll/onnxruntime_providers_shared.dll` |
| `tensorrt/` | TensorRT 10.16.0.72，供 C++ FP32 engine 使用 | [NVIDIA TensorRT 下载页](https://developer.nvidia.com/tensorrt)；Windows 选 10.16 对应的 x64 ZIP | `include/NvInferRuntime.h` 等头文件、`dll/nvinfer_10.dll`、`dll/nvinfer_plugin_10.dll`；当前工程动态加载 DLL，不链接 TensorRT `.lib` |
| `opencv/` | OpenCV 4.5.5，供 C++ 图像读写与预后处理 | [OpenCV 4.5.5 Release](https://github.com/opencv/opencv/releases/tag/4.5.5)；Windows 预编译包或用 VS2019 构建 | `include/opencv2/...`、`lib/opencv_world455.lib`、`lib/opencv_world455d.lib`、`dll/opencv_world455.dll`、`dll/opencv_world455d.dll` |
| `cuda/` | CUDA Runtime 12.8 DLL，供当前 TensorRT 推理路径使用 | [PyTorch CUDA 12.8 wheel](https://download.pytorch.org/whl/cu128) 中的 `torch/lib/cudart64_12.dll`；也可从匹配的 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 获取 | `dll/cudart64_12.dll` |

C++ 编译另需 `cuda_runtime_api.h`：本机位于 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`。可从 [NVIDIA CUDA Toolkit 11.8 下载页](https://developer.nvidia.com/cuda-11-8-0-download-archive) 获取。CMake 用 `-DCUDA_INCLUDE_DIR=...` 覆盖路径；VS2019 用 MSBuild 属性 `CudaIncludeDir` 覆盖。当前运行 DLL 来自 CUDA 12.8，头文件来自已安装的 11.8 工具包；本工程仅调用稳定的 CUDA Runtime 基础接口，已在本机编译与运行验证。跨机器部署仍应重新验证驱动、CUDA、TensorRT 与 GPU 组合。

历史实验使用的 `onnxruntime/` 目录是 1.30.0；默认 C++ 构建选用 `onnxruntime_1_15_1/`，Python 的 `requirements.txt` 使用 1.30.0。不要混放不同版本的头文件、导入库与 DLL。用户消息中的“1.51.1”按给定目录实际版本 1.15.1 落地。

Python 大包由 `setup.ps1` 安装到忽略的 `.venv/`：PyTorch 2.11.0+cu128 来自 [PyTorch CUDA 12.8 wheel 索引](https://download.pytorch.org/whl/cu128)，TensorRT Python 10.16.0.72 的安装方式见 [NVIDIA TensorRT 安装说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/installing.html)，YOLO26 权重由 [Ultralytics 模型包](https://docs.ultralytics.com/models/yolo26) 首次运行时获取。`uv` 安装方式见 [官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

依赖目录结构：

```text
third_party/
  onnxruntime_1_15_1/{include,lib,dll}/
  tensorrt/{include,lib,dll}/
  opencv/{include,lib,dll}/
  cuda/dll/
```

TensorRT `.engine` 与构建时的 TensorRT、GPU 架构及运行环境有关。模型权重和 engine 不进入 Git；换机器后从 `.pt` 重新导出。
