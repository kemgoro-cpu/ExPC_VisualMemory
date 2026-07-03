param(
    [string]$Python = "python",
    [string]$YomiEnvironment = ".venv-yomi-gpu",
    [string]$PaddleEnvironment = ".venv-paddle-gpu"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
    $ProjectRoot = $PSScriptRoot
}
Push-Location $ProjectRoot
try {
$PackageTarget = ".[dev]"
$Wheel = Get-ChildItem (Join-Path $ProjectRoot "packages\external_pc_visual_memory-*.whl") -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Wheel) { $PackageTarget = $Wheel.FullName }
& $Python -m venv $YomiEnvironment
$YomiPython = Join-Path $YomiEnvironment "Scripts\python.exe"
& $YomiPython -m pip install --upgrade pip
& $YomiPython -m pip install torch==2.12.1+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
if ($Wheel) {
    & $YomiPython -m pip install "$($Wheel.FullName)[yomitoku]"
}
else {
    & $YomiPython -m pip install -e ".[dev,yomitoku]"
}
& $YomiPython -c "import torch; assert torch.cuda.is_available(); print('YomiToku Torch CUDA:', torch.__version__, torch.cuda.get_device_name(0))"

& $Python -m venv $PaddleEnvironment
$PaddlePython = Join-Path $PaddleEnvironment "Scripts\python.exe"
& $PaddlePython -m pip install --upgrade pip
if ($Wheel) {
    & $PaddlePython -m pip install $Wheel.FullName
}
else {
    & $PaddlePython -m pip install -e ".[dev]"
}
& $PaddlePython -m pip install "paddleocr>=3,<4"
& $PaddlePython -m pip install paddlepaddle-gpu==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
# Paddle 3.3.1 currently leaves nvJitLink unconstrained; 12.9 breaks its cuDNN 12.6 load on Windows.
& $PaddlePython -m pip install --force-reinstall nvidia-nvjitlink-cu12==12.6.85
& $PaddlePython -c "import paddle; assert paddle.device.is_compiled_with_cuda(); paddle.set_device('gpu:0'); print('Paddle CUDA:', paddle.__version__, paddle.device.get_device())"
}
finally {
    Pop-Location
}
