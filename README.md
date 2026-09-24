# YOLO26 训练与推理工程

Windows 下的 YOLO26 检测训练、CPU ONNX / TensorRT 推理与 C++17 性能测试。Python 与 C++ 都可输出标注图；C++ 工程可直接用 Visual Studio 2019 打开 [`cpp/YOLO26.sln`](cpp/YOLO26.sln)。本仓库只保存源码、工程配置和说明，数据、权重、engine、依赖及运行结果保留在本机。

## 目录

| 路径 | 用途 |
| --- | --- |
| `python/prepare_labelme.py` | 将带拍摄时间的 LabelMe BMP/JSON 转为 YOLO 数据集，按时间组隔离验证集 |
| `python/train.py` | 训练 YOLO26，自带状态 JSON 和预计剩余时间 |
| `python/dashboard.py`、`python/dashboard.html` | 本机网页看板，查看训练进度、ETA 和验证曲线 |
| `python/export.py` | 从 `best.pt` 导出 FP32、640、batch 1、免 NMS 的 `.engine` 与独立 `.onnx` |
| `python/infer.py` | Python 直连 ONNX Runtime CPU 或 TensorRT 推理、计时、保存检测图 |
| `python/experiments/` | 早期 YOLOv8n / YOLO26n 公开数据集对比脚本 |
| `cpp/src/detect.cpp` | C++ 完整检测：预处理、ONNX/TRT 推理、后处理和可选结果图 |
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

原始数据应为同名 `.bmp` 与 LabelMe `.json`，文件名中含 `15h57m52s433ms` 形式的拍摄时间。已验证的打结数据按 3 秒时间间隔分组，验证组为 `0,15,22`：

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

## VS2019 C++ 推理

1. 按 [`third_party/README.md`](third_party/README.md) 放置依赖。安装 VS2019 C++ x64 工具集与 Windows 10 SDK。
2. 直接打开 [`cpp/YOLO26.sln`](cpp/YOLO26.sln)，选择 `Release|x64` 或 `Debug|x64` 并生成解决方案。两种配置都启用 `/Zi` 和链接器 PDB。
3. 程序输出在 `x64/output/vs2019/<配置>/<项目名>/`。若 CUDA 头文件不在本机默认路径，在项目属性中覆盖 `CudaIncludeDir`。

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

当前 C++ 默认使用 ONNX Runtime 1.16.3，Python 使用 1.30.0，二者性能不能直接归因于语言差异。本机相同模型与 C++ 程序、20 次预热 / 128 次测量的两图复测中，1.16.3 约 33.2 ms，1.30.0 约 25.7 ms；需要比较语言速度时应统一运行时版本。

## Git 管理

`.gitignore` 排除 `third_party` 大依赖、`.venv`、`x64` 输出、数据图片、权重、ONNX/engine 和构建二进制。当前仓库不配置远程地址，也不推送。移机后按依赖说明下载并重新生成模型产物；原始工业数据由使用者单独管理。
