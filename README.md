# YOLO26 训练与推理工程

Windows 下的 YOLO26 检测训练、CPU ONNX / TensorRT 推理与 C++17 性能测试。Python 与 C++ 都可输出标注图；C++ 工程可直接用 Visual Studio 2019 打开 [`cpp/YOLO26.sln`](cpp/YOLO26.sln)。本仓库只保存源码、工程配置和说明，数据、权重、engine、依赖及运行结果保留在本机。

## 目录

| 路径 | 用途 |
| --- | --- |
| `python/prepare_labelme.py` | 首版纯 NG 实验使用的时间组转换脚本；当前 NG/OK 实验不用 |
| `python/prepare_knot_iteration.py` | 本次 NG/OK 数据按比例随机拆分，保留源图 SHA-256 清单 |
| `python/train.py` | 训练 YOLO26，自带状态 JSON 和预计剩余时间 |
| `python/dashboard.py`、`python/dashboard.html` | 本机网页看板，查看训练进度、ETA 和验证曲线 |
| `python/iteration_dashboard.py` | 汇总多轮训练进度、ETA 与可视化报告入口 |
| `python/evaluate_knot.py`、`python/predict_knot_pt.py` | 当前打结模型的逐图 NG/OK 评估与 PyTorch 结果图 |
| `python/evaluate_knot_cpp_onnx.py` | 核对 C++ 全量预测、严格 IoU 检出和 Python 逐框一致性 |
| `python/visualize_knot_features.py`、`python/probe_knot_occlusion.py` | 关键卷积层热图与区域遮挡诊断 |
| `python/probe_brightness_shortcut.py` | 检查采集亮度是否成为类别捷径 |
| `python/export.py` | 从 `best.pt` 导出 FP32、640、batch 1、免 NMS 的 `.engine` 与独立 `.onnx` |
| `python/export_knot_onnx.py` | 导出 V6 同口径的动态 one-to-many 单类别 ONNX |
| `python/infer.py` | Python 直连 ONNX Runtime CPU 或 TensorRT 推理、计时、保存检测图 |
| `python/experiments/` | 早期 YOLOv8n / YOLO26n 公开数据集对比脚本 |
| `cpp/src/detect.cpp` | C++ 完整检测：预处理、ONNX/TRT 推理、后处理和可选结果图 |
| `cpp/include/vt_detect_api.h`、`cpp/include/vt_detect_version.h` | 通用检测接口、结果结构及版本常量 |
| `cpp/src/vt_detect.cpp`、`cpp/src/vt_detect_version.rc`、`cpp/vt_detect.vcxproj` | 独立推理 DLL，负责模型加载、预处理、ONNX 执行、后处理和 Windows 文件版本 |
| `cpp/src/knot_onnx.cpp`、`cpp/KnotOnnxDetect.vcxproj` | 通过 `LoadLibraryExW` 动态加载 DLL，批量保存叠框图和逐框 CSV |
| `cpp/src/main.cpp` | C++ 纯运行时测速，输入预处理好的 `1x3x640x640` FP32 `.bin` |
| `cpp/YOLO26.sln`、`cpp/CMakeLists.txt` | VS2019 原生解决方案与 CMake 两种构建入口 |
| `third_party/README.md` | 各 C++ 大依赖的版本、官方地址和放置位置 |
| `x64/output/` | 本机数据集、训练、导出、构建、推理结果；不进 Git |

## Python 环境

在仓库根目录用 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
$py = '.\.venv\Scripts\python.exe'
```

`setup.ps1` 使用 Python 3.12、PyTorch 2.11.0+cu128 和根目录 [`requirements.txt`](requirements.txt)。已固定 Ultralytics 8.4.160、Python ONNX Runtime 1.30.0、TensorRT 10.16.0.72。预训练 `yolo26n.pt` 首次由 Ultralytics 自动下载；[YOLO26 官方说明](https://docs.ultralytics.com/models/yolo26)列出了模型名和下载方式。

### 1. 数据转换与训练

原始数据应为同名 `.bmp` 与 LabelMe `.json`。下面是首版纯 NG 训练时使用的时间组示例；本次加入 OK 后的比例随机拆分见第 3 节：

```powershell
$run = 'x64\output\my_knot_run'
& $py python\prepare_labelme.py --source 'F:\你的原始数据目录' --output-dir $run --class-name 打结 --val-groups 0,15,22
& $py python\train.py --data "$run\data.yaml" --output-dir $run --name yolo26n_knot --epochs 100 --batch 8
```

另开一个 PowerShell 窗口启动看板：

```powershell
& '.\.venv\Scripts\python.exe' python\dashboard.py --run-dir 'x64\output\my_knot_run' --name yolo26n_knot
```

浏览器打开 `http://127.0.0.1:8765/`。状态文件在 `<run>/training_status.json`，权重在 `<run>/runs/<name>/weights/best.pt`。如已有 YOLO 数据集，可直接把其 `data.yaml` 传给 `train.py`。

### 2. 导出与 Python 推理

先导出 TensorRT engine，再单独导出 CPU ONNX；YOLO26 使用 `nms=False` 的 `(1,300,6)` 分支。`export.py` 已按此顺序执行：

```powershell
$export = 'x64\output\my_export'
& $py python\export.py --weights "$run\runs\yolo26n_knot\weights\best.pt" --output-dir $export
& $py python\infer.py onnx "$export\yolo26n_e2e.onnx" "$run\dataset\images\val" --output-dir 'x64\output\infer_onnx' --warmup 20 --iterations 1
& $py python\infer.py trt "$export\yolo26n_e2e.engine" "$run\dataset\images\val" --output-dir 'x64\output\infer_trt' --warmup 20 --iterations 1
```

`infer.py` 仅针对上述 FP32、640、免 NMS 导出格式；图片和 `inference.json` 存在指定输出目录。计时包含预处理、运行时和后处理，不包含读图、绘图、写盘与模型加载。要测单张图的稳定耗时，可将来源设为单张图片并使用 `--iterations 128`。

### 3. 口罩打结 NG/OK 微调与 PyTorch 逐图验证

含 OK 空标注的 LabelMe 数据使用 `python/prepare_knot_iteration.py` 按 NG/OK 比例随机拆成训练/验证集，不按拍摄日期分组。训练命令可调整优化器、学习率、水平翻转、框损失和亮度增强。`python/evaluate_knot.py` 同时统计 IoU≥0.5 的 NG 检出与 OK 图像误报，`python/visualize_knot_features.py` 导出卷积层热图。`python/predict_knot_pt.py` 可对任意单图或目录保存 PyTorch 推理叠图：

```powershell
$run = 'x64\output\20260928_084600'
$best = "$run\experiments\v6_box10\runs\yolo26n_knot\weights\best.pt"
& $py python\predict_knot_pt.py --weights $best --source "$run\dataset\images\val" --output-dir "$run\predict\v6_raw_val" --confidence 0.25 --imgsz 640
```

新数据按比例随机拆分的示例：

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$newRun = "x64\output\$stamp"
& $py python\prepare_knot_iteration.py --source 'F:\你的原始数据目录' --output-dir $newRun --val-ratio 0.25 --seed 42
```

平均灰度归一与 `--blur-kernel 5` 是光照/纹理诊断选项，需在独立批次确认后再用于正式推理。本次数据审计、训练轮次、光照扰动、逐图漏检归因及验收限制见 [`docs/260928口罩打结检测迭代记录.md`](docs/260928口罩打结检测迭代记录.md)。可视化报告在本机 `x64/output/20260928_084600/口罩打结检测可视化.html`；多轮训练进度看板可运行 `python/iteration_dashboard.py --run-dir x64/output/20260928_084600`。

## VS2019 C++ 推理

1. 按 [`third_party/README.md`](third_party/README.md) 放置依赖。安装 VS2019 C++ x64 工具集与 Windows 10 SDK。
2. 直接打开 [`cpp/YOLO26.sln`](cpp/YOLO26.sln)，选择 `Release|x64` 或 `Debug|x64` 并生成解决方案。两种配置都启用 `/Zi` 和链接器 PDB。
3. 程序输出在 `x64/output/vs2019/<配置>/<项目名>/`。若 CUDA 头文件不在本机默认路径，在项目属性中覆盖 `CudaIncludeDir`。

打结模型作为 `vt_detect.dll` 的首个检测模块推理，`KnotOnnxDetect.exe` 通过 `LoadLibraryExW` 和 `GetProcAddress` 查找导出函数，不链接 ONNX Runtime 导入库。两个工程均包含在解决方案中；构建 `KnotOnnxDetect` 时也会构建 DLL。部署时将 `vt_detect.dll`、ONNX Runtime 1.15.1 的 `onnxruntime.dll` 与 `onnxruntime_providers_shared.dll`、OpenCV 4.5.5 的 `opencv_world455.dll` 放在调用程序旁，并提供模型文件。Debug 配置使用 `opencv_world455d.dll`。部署文件由本机 `third_party/` 生成，不纳入 Git。

公共接口见 [`cpp/include/vt_detect_api.h`](cpp/include/vt_detect_api.h)，动态加载和显式 `Destroy` 示例见 [`cpp/src/knot_onnx.cpp`](cpp/src/knot_onnx.cpp)。加载后先通过 `VTDetect_GetAbiVersion` 核对 ABI 版本 1；`VTDetect_Create(modelPath)` 只需 UTF-16 模型路径，默认 CPU 4 线程、置信度 0.25、NMS IoU 0.5、最多 100 框。需要调整这些值时使用可选的 `VTDetect_CreateEx`。`VTDetect_Detect(handle, &image)` 直接接收 `CV_8UC3` 的 `cv::Mat`，DLL 读取尺寸和行跨度并返回内部持有的框与分段耗时；返回空指针时使用 `VTDetect_LastError` 查询 UTF-8 错误。结果在同一句柄下次检测或 `VTDetect_Destroy` 前有效，每个工作线程应创建自己的句柄。调用端可按以下顺序使用动态加载得到的函数指针：

```cpp
VTDetectHandle handle = api.create(modelPath.c_str());
const VTDetectResult* result = api.detect(handle, &image);
// 在下一次检测前读取 result->boxes 和 result->box_count。
api.destroy(handle);
handle = nullptr;
```

`api` 的 `LoadLibraryExW`、`GetProcAddress`、版本校验和异常路径释放见上述完整示例。当前软件版本为 `1.0.0`，Windows DLL 文件版本为 `1.0.0.0`。这次通用接口替代了旧的 `KnotOnnx_*` 导出，旧调用程序需按新头文件重新编译。`cv::Mat` 跨 DLL 传递要求调用端与 DLL 使用相同的 OpenCV 4.5.5、x64 和兼容的 MSVC 运行库；当前仅支持动态 FP32、单类别 `[1,5,anchors]` 输出模型，后续模型类型需要在 DLL 中实现对应解码。

完整检测命令：

```powershell
$exe = 'x64\output\vs2019\Release\YoloCppDetect\YoloCppDetect.exe'
& $exe onnx 'x64\output\my_export\yolo26n_e2e.onnx' 'x64\output\my_knot_run\dataset\images\val\sample.bmp' 4 20 128 'x64\output\result_onnx.png'
& $exe trt 'x64\output\my_export\yolo26n_e2e.engine' 'x64\output\my_knot_run\dataset\images\val\sample.bmp' 4 20 128 'x64\output\result_trt.png'
```

参数依次为后端、模型、图片、CPU 线程数、预热次数、测量次数、可选结果图路径。最后一个参数省略时只输出耗时和框数。完整检测计时不包含读图、绘图、写盘与模型加载。`YoloCppBench.exe` 则只测模型运行时，输入必须是 640×640、CHW、FP32 的 `.bin`。

也可用 CMake 构建：

```powershell
cmake -S cpp -B x64/output/cmake-build -G 'Visual Studio 16 2019' -A x64
cmake --build x64/output/cmake-build --config Release
```

只构建打结 DLL 与动态加载演示程序时，可在 CMake 配置中加入 `-DYOLO26_BUILD_LEGACY_TRT_TOOLS=OFF`，此时无需历史 TensorRT/CUDA 工具依赖。

当前 C++ 默认使用 ONNX Runtime 1.15.1，来自本机 RapidOCR 的 Windows x64 CPU 包；Python 使用 1.30.0，二者性能不能直接归因于语言差异。此前 1.16.3 与 1.30.0 的计时属于历史实验，不能充当当前 1.15.1 的结果。

### V6 打结模型的 C++ ONNX 全量验证

V6 的既有 Python 验证使用 one-to-many + NMS。`python/export.py` 导出的 `nms=False` 是另一个检测头，不能沿用 V6 的阈值和检出率；本次使用动态输入、单类别原始输出 `[1,5,anchors]`，由 C++ 以 0.25 置信度和 0.5 NMS IoU 后处理。对单张裁剪图使用 640 尺寸的最小矩形补边，通常实际张量为 352×640。

```powershell
$py = '.\.venv\Scripts\python.exe'
$run = 'x64\output\20260928_170714'
$data = 'x64\output\20260928_113627\expanded'
$best = 'x64\output\20260928_084600\experiments\v6_box10\runs\yolo26n_knot\weights\best.pt'
& $py python\export_knot_onnx.py --weights $best --output-dir "$run\models" --name v6_knot
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe' cpp\YOLO26.sln /t:KnotOnnxDetect /p:Configuration=Release /p:Platform=x64
$verify = 'x64\output\20260929_085458'
& 'x64\output\vs2019\Release\KnotOnnxDetect\KnotOnnxDetect.exe' "$run\models\v6_knot.onnx" "$data\dataset\images" "$verify\vs2019" 4
& $py python\evaluate_knot_cpp_onnx.py --manifest "$data\dataset_manifest.json" --predictions "$verify\vs2019\predictions.csv" --image-root "$data\dataset\images" --output-dir "$verify\evaluation"
```

本机通用接口全量验证结果：185 张图中 NG 严格检出 133/134，OK 误报 0/51；其中训练划分 NG 105/105、OK 0/40，验证划分 NG 28/29、OK 0/11。新增 30 张 PNG OK 没参与 V6 训练。185 张图片与 F 盘源目录逐张 SHA-256 一致，但训练划分和验证划分不能混作独立测试；验证集此前也参与选型。逐图清单、汇总和错误图在 `$verify\evaluation`，全图叠框在 `$verify\vs2019\overlays`，本机部署包在 `$verify\deploy`。新旧接口的 196 行逐框预测完全一致。本次 VS2019 Release、CPU 4 线程在 10 次预热后的 ONNX 模型执行中位数约 16.96 ms，计时不含读图、叠图和写盘；CPU 并发负载会影响绝对耗时。

## Git 管理

`.gitignore` 排除 `third_party` 大依赖、`.venv`、`x64` 输出、数据图片、权重、ONNX/engine 和构建二进制。当前仓库不配置远程地址，也不推送。移机后按依赖说明下载并重新生成模型产物；原始工业数据由使用者单独管理。
