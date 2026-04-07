# C# YOLO CUDA 接口文档

## 版本与范围
本文定义 `EvanWu.YoloCuda` 首版公共 API 契约。

首版范围：
- 运行环境：Windows x64 + .NET 10 LTS。
- GPU 后端：NVIDIA CUDA，通过 ONNX Runtime CUDA Execution Provider。
- 模型类型：YOLOv8/YOLO11 ONNX object detection 模型。
- 输入类型：单张图片路径。
- 输出类型：原图坐标系下的目标检测框列表。

首版不提供：
- CPU fallback。
- DirectML/WinML 后端。
- 视频流、批量推理、异步推理 API。
- YOLO 分割、姿态估计、旋转框 OBB 或分类接口。

## NuGet 与依赖契约
类库实现依赖：
- `Microsoft.ML.OnnxRuntime.Gpu`：创建 CUDA `InferenceSession` 并执行 ONNX 推理。
- `SixLabors.ImageSharp`：读取图片、缩放、像素转换和预处理。

调用方运行时必须确保：
- NVIDIA 驱动可用。
- ONNX Runtime GPU 所需 CUDA/cuDNN DLL 能被进程加载。
- 模型文件和标签文件路径对当前进程可读。

CUDA provider 初始化失败时，首版必须抛出明确异常，不允许静默降级 CPU。

## 命名空间
公共 API 统一放在：

```csharp
namespace EvanWu.YoloCuda;
```

## YoloDetectorOptions
```csharp
public sealed record YoloDetectorOptions
{
    public required string ModelPath { get; init; }
    public string? LabelsPath { get; init; }
    public int InputWidth { get; init; } = 640;
    public int InputHeight { get; init; } = 640;
    public int GpuDeviceId { get; init; } = 0;
    public float ConfidenceThreshold { get; init; } = 0.25f;
    public float NmsThreshold { get; init; } = 0.45f;
}
```

字段契约：

| 字段 | 必填 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `ModelPath` | 是 | 无 | 非空且文件存在 | ONNX 模型路径。 |
| `LabelsPath` | 否 | `null` | 为 `null` 或文件存在 | 标签文件路径；未提供时使用 `class_{ClassId}`。 |
| `InputWidth` | 否 | `640` | `> 0` | 模型输入宽度。 |
| `InputHeight` | 否 | `640` | `> 0` | 模型输入高度。 |
| `GpuDeviceId` | 否 | `0` | `>= 0` | CUDA GPU 设备编号。 |
| `ConfidenceThreshold` | 否 | `0.25f` | `[0, 1]` | 最低置信度阈值。 |
| `NmsThreshold` | 否 | `0.45f` | `[0, 1]` | NMS IoU 阈值。 |

路径规则：
- 相对路径按当前进程工作目录解析。
- 路径校验在构造 `YoloDetector` 时执行。
- 标签文件按 UTF-8 文本读取，每行一个类别名，行号从 `0` 开始对应 `ClassId`。

## YoloDetector
```csharp
public sealed class YoloDetector : IDisposable
{
    public YoloDetector(YoloDetectorOptions options);

    public IReadOnlyList<DetectionResult> Detect(string imagePath);

    public void Dispose();
}
```

构造函数职责：
- 校验 `YoloDetectorOptions`。
- 读取并缓存标签文件。
- 创建 ONNX Runtime CUDA `InferenceSession`。
- 校验模型输入/输出形状是否符合首版 YOLO detection 预期。

`Detect(string imagePath)` 契约：
- `imagePath` 必须非空且指向可读取图片文件。
- 图片会按 letterbox 方式缩放到 `InputWidth` x `InputHeight`。
- 输入张量格式为 `float32`、RGB、NCHW、归一化到 `[0, 1]`。
- 返回值为不可变语义的检测结果列表。
- 无检测结果时返回空列表，不抛异常。
- 返回结果按 `Confidence` 从高到低排序。

生命周期契约：
- `YoloDetector` 持有 ONNX Runtime session，调用方应使用 `using` 或显式调用 `Dispose()`。
- `Dispose()` 后继续调用 `Detect()` 应抛出 `ObjectDisposedException`。

线程安全契约：
- 首版不承诺同一个 `YoloDetector` 实例可被多个线程同时调用。
- 多线程调用方应为每个工作线程创建独立实例，或在外部加锁。

## DetectionResult
```csharp
public sealed record DetectionResult(
    int ClassId,
    string Label,
    float Confidence,
    BoundingBox Box);
```

字段契约：
- `ClassId`：模型输出类别索引，必须大于等于 `0`。
- `Label`：类别名称；如果未提供标签文件或类别越界，使用 `class_{ClassId}`。
- `Confidence`：最终置信度，范围 `[0, 1]`。
- `Box`：原图坐标系下的边界框。

排序规则：
- `Detect()` 返回结果按 `Confidence` 降序排列。
- 同等置信度下不保证稳定排序。

## BoundingBox
```csharp
public readonly record struct BoundingBox(
    float X,
    float Y,
    float Width,
    float Height);
```

坐标语义：
- `X` / `Y` 表示原图坐标系下左上角。
- `Width` / `Height` 表示原图坐标系下的宽高。
- 单位为像素。
- 输出前必须裁剪到原图边界内。
- 首版允许 `float` 小数坐标，由上层决定是否取整绘制。

## 模型 I/O 约定
首版支持的 ONNX 输入：
- 单输入张量。
- 默认形状语义为 `[1, 3, InputHeight, InputWidth]`。
- 数据类型为 `float32`。
- 色彩顺序为 RGB。

首版支持的 YOLO 输出：
- YOLOv8/YOLO11 detection head 导出的常见单输出张量。
- 后处理应能识别常见输出排列，并在无法确认时抛出 `NotSupportedException`。
- 框坐标解析后必须基于 letterbox 元数据还原到原图坐标。

如果模型包含动态输入尺寸，首版仍以 `YoloDetectorOptions.InputWidth` 和 `InputHeight` 作为实际推理尺寸。

## 错误处理契约
| 场景 | 异常类型 | 说明 |
| --- | --- | --- |
| `options` 为 `null` | `ArgumentNullException` | 构造函数立即失败。 |
| `ModelPath` 为空 | `ArgumentException` | 路径必须有值。 |
| `ModelPath` 文件不存在 | `FileNotFoundException` | 错误信息包含模型路径。 |
| `LabelsPath` 文件不存在 | `FileNotFoundException` | 仅当 `LabelsPath` 非空时校验。 |
| 输入尺寸小于等于 0 | `ArgumentOutOfRangeException` | 包含字段名。 |
| 阈值不在 `[0, 1]` | `ArgumentOutOfRangeException` | 包含字段名。 |
| `GpuDeviceId` 小于 0 | `ArgumentOutOfRangeException` | 包含字段名。 |
| CUDA provider 初始化失败 | `InvalidOperationException` | 包含 GPU device ID 和 ONNX Runtime 原始错误。 |
| 模型输入/输出形状不支持 | `NotSupportedException` | 包含实际输入/输出 shape。 |
| 图片路径为空 | `ArgumentException` | `Detect()` 立即失败。 |
| 图片文件不存在 | `FileNotFoundException` | 错误信息包含图片路径。 |
| 图片无法解码 | `InvalidDataException` | 包含 ImageSharp 或底层异常信息。 |
| 对已释放实例调用检测 | `ObjectDisposedException` | 指向 `YoloDetector`。 |

## 基础示例
```csharp
using EvanWu.YoloCuda;

using var detector = new YoloDetector(new YoloDetectorOptions
{
    ModelPath = "models/yolo11n.onnx",
    LabelsPath = "models/coco.names",
    InputWidth = 640,
    InputHeight = 640,
    GpuDeviceId = 0,
    ConfidenceThreshold = 0.25f,
    NmsThreshold = 0.45f
});

IReadOnlyList<DetectionResult> results = detector.Detect("samples/images/test.jpg");

foreach (DetectionResult result in results)
{
    Console.WriteLine($"{result.Label} {result.Confidence:P1} {result.Box}");
}
```

## 无标签文件示例
```csharp
using var detector = new YoloDetector(new YoloDetectorOptions
{
    ModelPath = "models/yolo11n.onnx"
});

IReadOnlyList<DetectionResult> results = detector.Detect("samples/images/test.jpg");
// Label 将回退为 class_0、class_1 等格式。
```

## 异常处理示例
```csharp
try
{
    using var detector = new YoloDetector(new YoloDetectorOptions
    {
        ModelPath = "models/yolo11n.onnx",
        GpuDeviceId = 0
    });

    IReadOnlyList<DetectionResult> results = detector.Detect("samples/images/test.jpg");
}
catch (InvalidOperationException ex)
{
    Console.Error.WriteLine($"CUDA initialization failed: {ex.Message}");
}
catch (NotSupportedException ex)
{
    Console.Error.WriteLine($"Unsupported ONNX model shape: {ex.Message}");
}
```

## 测试契约映射
后续实现时至少覆盖：
- `YoloDetectorOptions` 非法字段抛出预期异常。
- 配置了不存在的标签文件路径时按契约抛错；未提供标签文件时回退 `class_{ClassId}`。
- `Detect()` 输入图片不存在时抛出 `FileNotFoundException`。
- 伪造 YOLO 输出经过后处理后按置信度降序返回。
- NMS 对重叠框只保留最高置信度结果。
- letterbox 还原后的 `BoundingBox` 裁剪到原图边界。
- GPU 集成测试通过 `Category=Gpu`、`EVANWU_YOLO_ENABLE_GPU_TESTS=1` 和 `EVANWU_YOLO_TEST_MODEL` 显式运行，不依赖默认 `dotnet test` 环境。
