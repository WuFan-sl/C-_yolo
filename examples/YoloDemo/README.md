# YOLO Demo 使用说明

## 目录结构

```
YoloDemo/
├── Program.cs           # 主程序
├── YoloDemo.csproj      # 项目文件
├── convert.py           # .pt 转 .onnx 转换脚本
├── imgs/                # 放置待检测图片（不支持子文件夹）
├── output/              # 输出的渲染结果图
└── README.md            # 本文件
```

## 前置准备

### 1. 模型文件

在 `./models/` 目录下放置 **ONNX 格式** 的 YOLO 模型：

- `yolo11n.onnx` - YOLO11n 模型（或其他 YOLOv8/YOLO11 系列）
- `coco.names` 或 `labels.txt` - 类别标签文件（可选，会自动扫描匹配）

**自动模型匹配规则：**
- 优先使用 `.onnx` 文件
- 若只有 `.pt` 文件，会提示运行转换脚本
- 自动查找 `coco.names`, `labels.txt`, `classes.txt` 等标签文件

### 2. .pt 模型转换

如有 PyTorch `.pt` 模型，先运行转换脚本：

```bash
# 安装依赖
pip install ultralytics

# 转换 ./models 目录下所有 .pt 文件
python convert.py

# 或指定单个模型
python convert.py --model ./models/yolo11n.pt
```

### 3. 图片

将待检测的图片放入 `imgs/` 目录
- 支持的格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
- 不支持嵌套子文件夹

## 运行方式

```bash
dotnet run --project examples/YoloDemo
```

## 输出

### 控制台输出

每张图片的检测详情：
- **Inference Time** - 推理耗时（毫秒）
- **Detections** - 检测到的目标数量
- **Class** - 类别名称和 ID
- **Confidence** - 置信度（百分比）
- **Box** - 边界框坐标（X, Y, Width, Height）
- **OBB** - 当模型 metadata 为 `task=obb` 时，输出旋转框中心、宽高和弧度角

### 渲染输出

- **output/ 目录** - PNG 格式的渲染结果图
- 绿色边界框（2px）；OBB 模型会绘制旋转框
- 黑色半透明背景 + 绿色文字标签

## 输出文件名规则

原文件名 + `_result` 后缀

例如：`image1.png` → `image1_result.png`
