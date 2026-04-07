# C# YOLO CUDA 数据模型文档

## 配置模型
`YoloDetectorOptions` 是公共配置入口，必须能完整描述一次 detector 初始化所需的模型路径、标签路径、输入尺寸、GPU 设备和阈值。

字段约束：
- `ModelPath`：必填，指向 ONNX 文件。
- `LabelsPath`：可选，文本文件每行一个类别名；未提供时标签可回退为 `class_{ClassId}`。
- `InputWidth` / `InputHeight`：默认 `640`，必须为正整数。
- `GpuDeviceId`：默认 `0`，必须大于等于 `0`。
- `ConfidenceThreshold`：默认 `0.25`，取值范围 `[0, 1]`。
- `NmsThreshold`：默认 `0.45`，取值范围 `[0, 1]`。

## 检测结果模型
`DetectionResult` 表示一个目标检测框：
- `ClassId`：模型输出的类别索引。
- `Label`：类别名称。如果未提供标签文件，则使用 `class_{ClassId}`。
- `Confidence`：最终置信度，范围 `[0, 1]`。
- `Box`：原图坐标系下的边界框。
- `OrientedBox`：可选旋转框；仅当模型 metadata 标记 `task=obb` 且输出包含角度通道时返回。

## 边界框模型
`BoundingBox` 使用左上角坐标和宽高：
- `X`：左上角横坐标。
- `Y`：左上角纵坐标。
- `Width`：边界框宽度。
- `Height`：边界框高度。

所有输出框必须映射回原始图像坐标，而不是模型输入尺寸坐标。

## 旋转框模型
`OrientedBoundingBox` 使用中心点、宽高和旋转弧度：
- `CenterX` / `CenterY`：原图坐标系下的旋转框中心点。
- `Width` / `Height`：旋转框宽高。
- `RotationRadians`：模型输出的旋转角，单位为弧度。

当 `DetectionResult.OrientedBox` 有值时，`DetectionResult.Box` 仍提供该旋转框的外接水平框，便于现有调用方和 NMS 逻辑兼容。

## 内部预处理元数据
预处理阶段需要保留 letterbox 相关信息：
- 原图宽高。
- 模型输入宽高。
- 缩放比例。
- 水平和垂直 padding。

这些字段用于后处理阶段把模型坐标还原到原图坐标。

## 标签文件格式
标签文件采用纯文本格式：
```text
person
bicycle
car
...
```

规则：
- 每行一个标签。
- 行号从 `0` 开始对应 `ClassId`。
- 空行在读取时应忽略或显式报错；首版推荐忽略首尾空白，但不重排有效行。
