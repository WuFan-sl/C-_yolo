# C# YOLO CUDA 测试策略

## 测试目标
测试体系需要同时覆盖“无 GPU 可运行的确定性算法测试”和“Windows NVIDIA 环境中的真实 CUDA 集成测试”。默认测试必须适合普通开发机和 CI；GPU 测试必须显式开启。

## 单元测试
单元测试不依赖真实 ONNX 模型或 GPU，覆盖：
- `YoloDetectorOptions` 参数校验。
- 标签文件读取和类别映射。
- letterbox 缩放比例、padding 和输出尺寸。
- RGB 到 CHW float 张量转换。
- YOLO 输出张量解析。
- confidence 阈值过滤。
- NMS 去重。
- 坐标从模型输入尺寸还原到原图坐标。

默认命令：
```bash
dotnet test
```

## GPU 集成测试
GPU 集成测试依赖：
- Windows x64。
- NVIDIA GPU 和驱动。
- ONNX Runtime GPU 所需 CUDA/cuDNN DLL。
- 本地存在测试 ONNX 模型和测试图片。

建议使用 xUnit Trait：
```csharp
[Trait("Category", "Gpu")]
```

运行命令：
```bash
EVANWU_YOLO_ENABLE_GPU_TESTS=1 EVANWU_YOLO_TEST_MODEL=/path/to/model.onnx dotnet test --filter Category=Gpu
```

## 测试资产策略
- 小图片可以提交到 `tests/EvanWu.YoloCuda.Tests/TestAssets/images/`。
- 大型 ONNX 模型默认不提交。
- GPU 集成测试通过 `EVANWU_YOLO_ENABLE_GPU_TESTS=1` 显式开启，并通过 `EVANWU_YOLO_TEST_MODEL` 指定模型路径。
- 如果未设置显式开启变量或模型路径缺失，当前 GPU 测试不初始化 detector，避免默认测试依赖 GPU。

## 验收测试场景
- 输入一张包含已知目标的图片，返回至少一个稳定检测结果。
- 使用不存在的模型路径，返回明确文件错误。
- 使用非法阈值，返回参数错误。
- CUDA 初始化失败时，不静默降级 CPU，而是返回明确诊断。
- 多个高重叠框输入 NMS 后只保留最高置信度框。
