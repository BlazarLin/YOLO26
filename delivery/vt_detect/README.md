<!-- 创建时间：2026-09-30；功能：vt_detect 第三方交付说明；目的：让 VS2019 x64 调用方直接编译、运行和核对样例。 -->
# vt_detect 1.0.0 第三方交付包

本目录可整体复制给第三方。当前模型为口罩下成像打结检测 V6，ONNX Runtime 1.15.1 CPU 推理，OpenCV 4.5.5，Windows x64 / VS2019。`model/v6_knot.onnx` 与仓库 `models/v6_knot.onnx` 字节一致；模型输出为单类别 `[1,5,anchors]`，由 DLL 执行 NMS。

## 目录

- `include/vt_detect_api.h`：动态加载或导入库调用所需的公共接口。
- `include/vt_detect_version.h`：软件及文件版本号；`include/opencv2/`：匹配的 OpenCV 4.5.5 头文件。
- `model/v6_knot.onnx`：当前 C++ 推理模型。
- `sample/input/`：5 张输入样图；`sample/result/`：对应的 Release 叠框结果和 `predictions.csv`。
- `Debug/lib/`、`Release/lib/`：`vt_detect.lib` 与对应的 OpenCV 导入库。
- `Debug/dll/`、`Release/dll/`：`vt_detect.dll`、ONNX Runtime、对应的 OpenCV DLL，以及可直接运行的 `KnotOnnxDetect.exe`。
- `Debug/pdb/`、`Release/pdb/`：DLL 和演示程序的调试符号。
- `manifest.sha256`：主要模型、样图与二进制的 SHA-256 校验值。

## 先运行样例

在本目录打开 PowerShell：

```powershell
$root = (Get-Location).Path
& "$root\Release\dll\KnotOnnxDetect.exe" "$root\model\v6_knot.onnx" "$root\sample\input" "$env:TEMP\vt_detect_sample"
```

运行后查看 `%TEMP%\vt_detect_sample\overlays\input\` 中的叠框图和 `predictions.csv`。演示程序先对首图预热 10 次，再正式检测；它通过 `LoadLibraryExW` / `GetProcAddress` 加载同目录 `vt_detect.dll`，最后调用 `VTDetect_Destroy`。Debug 测试时把 `Release` 改为 `Debug`，并使用带 VS2019 调试运行库的开发机。

## 接入 VS2019

1. 选择 `x64`，C++17，配置名称为 `Debug` 或 `Release`。头文件目录添加本包 `include`。
2. 若直接链接导入库，库目录添加对应配置的 `lib`，附加依赖项添加 `vt_detect.lib` 与 `opencv_world455.lib`（Debug 为 `opencv_world455d.lib`）。若使用 `LoadLibraryExW` / `GetProcAddress` 动态加载，`vt_detect.lib` 无需链接；OpenCV 图像读写仍需对应的 OpenCV 库。
3. 将对应配置 `dll` 下的 DLL 放到调用程序 EXE 旁，或保留在可解析的 DLL 搜索目录。不要混用 Debug 与 Release 的 OpenCV。
4. 调用 `VTDetect_GetAbiVersion()`，确认返回 `1`；使用 `VTDetect_Create(L"模型完整路径")` 建模，向 `VTDetect_Detect(handle, &image)` 传入 `CV_8UC3` BGR `cv::Mat`，读取 `VTDetectResult`，最后调用 `VTDetect_Destroy(handle)`。结果指针只在该句柄下一次检测或销毁前有效；并发工作线程分别创建句柄。

默认参数为 CPU 4 线程、置信度 0.25、NMS IoU 0.5、最多 100 框。仅当需要调整时传 `VTDetectOptions` 调用 `VTDetect_CreateEx`。失败后用 `VTDetect_LastError()` 取当前线程 UTF-8 错误文本。接口和完整动态加载示例见仓库 `cpp/include/vt_detect_api.h` 与 `cpp/src/knot_onnx.cpp`。

公共接口含 `cv::Mat`，调用方必须使用兼容的 OpenCV 4.5.5 x64 C++ ABI。Release 运行机需要 Microsoft Visual C++ 2019 x64 可再发行组件。Debug 的 MSVC 调试运行库通常只在开发机提供。本包仅实现 CPU ONNX 推理，不包含 TensorRT。

## 模型验证范围

在原数据集 185 张图上的既有逐图验证：NG 133/134，OK 误检 0/51；其中独立验证集 NG 28/29、OK 误检 0/11，阈值 0.25、IoU 0.5。这个结果是该数据集上的离线验证，不等于产线长期检出率保证。本包 5 张样图的 Debug/Release 检测框一致，每张各检出 1 个框；具体置信度见 `sample/result/predictions.csv`。
