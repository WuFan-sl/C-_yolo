# YOLO Demo 使用说明

本示例用于验证 `EvanWu.YoloCuda` 在 Windows + .NET 环境中加载 YOLO ONNX 模型并对图片做目标检测。

当前代码支持：
- YOLOv8/YOLO11 普通 detection ONNX 输出。
- ONNX metadata 标记 `task=obb` 的 YOLO OBB 输出，demo 会绘制旋转框。
- 从 ONNX metadata 读取 `names` 类别名；如果额外提供 `labels.txt`，则优先使用外部标签文件。

## 目录结构

```text
examples/YoloDemo/
├── Program.cs
├── YoloDemo.csproj
├── convert.py
├── models/
│   ├── best.pt          # 本地 PyTorch 模型，不提交
│   ├── best.onnx        # 转换后的 ONNX 模型，不提交
│   ├── args.yaml        # 训练/导出参数，可用于确认 task/imgsz
│   └── labels.txt       # 可选，每行一个类别名
├── imgs/                # 待检测图片，不支持子文件夹
└── output/              # demo 输出图片
```

## 正确运行方式

`Program.cs` 使用相对路径 `./models`、`./imgs`、`./output`。因此最稳妥的运行方式是先进入 demo 目录：

```powershell
cd D:\000WuFan\112C#_yolo\examples\YoloDemo
dotnet run
```

也可以从仓库根目录指定项目，但必须同时把工作目录切到 demo 目录：

```powershell
cd D:\000WuFan\112C#_yolo\examples\YoloDemo
dotnet run --project .\YoloDemo.csproj
```

不推荐直接在仓库根目录运行 `dotnet run --project examples/YoloDemo`，因为程序会查找根目录下的 `./models` 和 `./imgs`，容易加载错模型或找不到图片。

如果要运行已编译的 exe：

```powershell
dotnet build .\YoloDemo.csproj
cd .\bin\Debug\net10.0-windows
.\YoloDemo.exe
```

此时需要确认 `bin\Debug\net10.0-windows\models` 和 `bin\Debug\net10.0-windows\imgs` 中有运行所需文件，或直接使用上面的 `dotnet run` 方式避免路径混乱。

## 模型准备

### 1. 放置模型

把模型放到 `examples/YoloDemo/models/`：

```text
examples/YoloDemo/models/best.pt
```

如果已有 ONNX：

```text
examples/YoloDemo/models/best.onnx
```

可选标签文件：

```text
examples/YoloDemo/models/labels.txt
```

`labels.txt` 每行一个类别名，例如：

```text
勺子
```

如果 ONNX metadata 中已有 `names={0: '勺子'}`，不放 `labels.txt` 也可以，demo 会自动使用 metadata 类别名。

### 2. 转换 `.pt` 到 `.onnx`

安装依赖：

```powershell
py -m pip install ultralytics
```

转换指定模型：

```powershell
cd D:\000WuFan\112C#_yolo\examples\YoloDemo
py .\convert.py --model .\models\best.pt
```

脚本会交互询问 `imgsz`。必须使用训练/导出时匹配的输入尺寸。当前 `args.yaml` 中是：

```text
task: obb
imgsz: 320
```

所以当前 `best.pt` 应使用 `imgsz=320` 转换。转换后的 `best.onnx` metadata 应包含：

```text
task=obb
imgsz=[320, 320]
names={0: '勺子'}
```

如果 metadata 缺少 `task=obb`，C# 后处理可能无法区分 OBB 输出和普通 detection 输出。

## 图片准备

把待检测图片放到：

```text
examples/YoloDemo/imgs/
```

支持格式：
- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

不支持递归扫描子文件夹。

## 预期输出

运行时会显示：

```text
Model: best.onnx
Labels File: (none; using ONNX metadata when available)
Requested Input Size: 640x640
Execution: CPU
Actual Input Size: 320x320
Model Task: obb
```

说明：
- `Requested Input Size` 是 demo 传入的默认配置。
- `Actual Input Size` 是从 ONNX 输入 shape 读取到的真实尺寸；当前模型是 `320x320`。
- `Model Task: obb` 表示 C# 已按 OBB 输出解析。
- `Execution: CPU` 表示 CUDA provider 没有成功加载，程序回退到 CPU；这影响速度，不应影响解析正确性。

每张图片的检测输出示例：

```text
Detections: 1
Class: 勺子 (ID: 0)
Confidence: 94.99%
Box: X=333.1, Y=50.1, W=879.7, H=679.6
OBB: CX=772.9, CY=390.0, W=841.1, H=301.4, Angle=0.520 rad
```

输出图片保存在：

```text
examples/YoloDemo/output/
```

输出文件名规则：

```text
原文件名 + _result.png
```

例如：

```text
1.jpg -> 1_result.png
```

普通 detection 模型会绘制水平框；OBB 模型会绘制旋转框。

## CUDA 运行说明

当前项目使用：

```text
Microsoft.ML.OnnxRuntime.Gpu.Windows 1.24.4
```

如果运行时出现类似错误：

```text
onnxruntime_providers_cuda.dll depends on cudnn64_9.dll which is missing
```

说明本机 CUDA/cuDNN 运行时 DLL 不完整或不在进程可加载路径中。此时 demo 会显示：

```text
Execution: CPU
```

处理方式：
- 安装与 ONNX Runtime GPU `1.24.4` 匹配的 NVIDIA CUDA/cuDNN 运行时。
- 确认 `cudnn64_9.dll` 等 DLL 能被进程加载，例如在系统 `PATH` 或应用运行目录可见。
- 重新运行 demo，确认 `Execution: CUDA`。

## 常见问题

### 1. 未转换的 `.pt` 效果好，ONNX demo 效果很差

先检查 `models/args.yaml` 和 ONNX metadata。当前模型是：

```text
task: obb
imgsz: 320
```

如果把 OBB 模型按普通 detection 解析，会出现大量错误框、`class_1`、置信度超过 100% 等现象。当前代码已支持 metadata 标记的 `task=obb` 输出，正确结果应显示 `Class: 勺子` 和 `OBB:` 行。

### 2. 显示 `Labels File: (none)`

这不一定是错误。若 ONNX metadata 中有 `names`，demo 会自动使用 metadata 类别名。

如果希望显式指定标签，可以添加：

```text
examples/YoloDemo/models/labels.txt
```

### 3. 显示 `Actual Input Size: 320x320`

这是正确的。当前 ONNX 模型输入 shape 是 `[1, 3, 320, 320]`，demo 会自动使用模型尺寸，而不是继续使用默认 `640x640`。

### 4. 想验证 ONNX metadata

可以用 ONNX Runtime 读取模型 metadata，确认至少包含：

```text
task=obb
imgsz=[320, 320]
names={0: '勺子'}
```

### 5. 输出框仍然偏大

OBB 模型会同时输出：
- `OBB`：真正用于绘制的旋转框。
- `Box`：旋转框的外接水平框，用于兼容普通检测结果和当前 NMS。

判断效果时应看输出图片中的绿色旋转框，不要只看控制台里的 `Box` 外接矩形。

## 本地文件提交规则

不要提交以下大文件或机器文件：
- `.pt`
- `.onnx`
- `models/`
- `imgs/`
- `output/`
- `.vs/`
- `bin/`
- `obj/`

只提交源码、README、测试和必要的小型配置文件。
