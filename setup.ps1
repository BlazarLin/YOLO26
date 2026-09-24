# 创建时间：2026-09-23；功能：创建隔离 Python 环境；目的：复现实验依赖。
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw '未找到 uv，请先安装 uv。'
}
if (-not (Test-Path -LiteralPath $python)) {
    uv venv (Join-Path $root '.venv') --python 3.12
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
uv pip install --python $python 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --link-mode=copy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv pip install --python $python -r (Join-Path $root 'requirements.txt') --link-mode=copy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -c "import torch,tensorrt,onnxruntime; assert torch.cuda.is_available(); assert tensorrt.Builder(tensorrt.Logger()); print(torch.__version__, tensorrt.__version__, onnxruntime.__version__)"
exit $LASTEXITCODE
