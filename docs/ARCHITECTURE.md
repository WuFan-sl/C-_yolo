# C# YOLO CUDA 类库架构文档

## 架构摘要
类库采用“薄公共 API + 可测试核心算法 + ONNX Runtime 适配层”的结构。公共调用方只接触 `YoloDetector`、配置对象和检测结果；图像预处理、张量推理、YOLO 后处理分别拆分，便于单元测试和后续替换实现。

## 目标结构
```text
src/
  EvanWu.YoloCuda/
    YoloDetector.cs
    YoloDetectorOptions.cs
    DetectionResult.cs
    BoundingBox.cs
    Preprocessing/
      ImagePreprocessor.cs
      LetterboxResult.cs
      PreprocessingResult.cs
    Postprocessing/
      YoloPostprocessor.cs
      NonMaxSuppression.cs
    Runtime/
      IYoloInferenceSession.cs
      OnnxCudaInferenceSession.cs
      OnnxCudaSessionFactory.cs
tests/
  EvanWu.YoloCuda.Tests/
    Preprocessing/
    Postprocessing/
    Runtime/
docs/
```

## 核心组件
- `YoloDetector`：公共入口，负责参数校验、模型会话生命周期、调用预处理/推理/后处理。
- `IYoloInferenceSession` / `OnnxCudaInferenceSession`：封装 ONNX Runtime `InferenceSession`，让 `YoloDetector` 能在单元测试中注入 fake session，避免默认测试依赖真实 GPU 或模型。
- `OnnxCudaSessionFactory`：封装 ONNX Runtime CUDA session 创建，统一处理 GPU 设备 ID、图优化和错误信息。
- `ImagePreprocessor`：读取图片、letterbox 缩放、归一化、RGB CHW 张量生成，并返回坐标还原所需元数据。
- `YoloPostprocessor`：解析 YOLOv8/YOLO11 检测输出，做置信度过滤、类别解析、坐标还原和 NMS。
- `NonMaxSuppression`：纯算法模块，不依赖 ONNX Runtime，必须有独立单元测试。

## 数据流
```text
Image path / image stream
  -> ImagePreprocessor
  -> DenseTensor<float> input
  -> ONNX Runtime CUDA InferenceSession
  -> Raw output tensor
  -> YoloPostprocessor
  -> IReadOnlyList<DetectionResult>
```

## GPU 策略
首版强制使用 CUDA/NVIDIA 后端。创建会话时应显式使用 CUDA provider；如果初始化失败，抛出包含模型路径、GPU device ID 和可能依赖问题的异常。首版不静默降级到 CPU，避免误判性能和部署状态。

## 扩展点
- 后续可通过新增 runtime factory 支持 DirectML，但不改变 `YoloDetector` 的主要使用方式。
- 后续可新增 `IYoloPostprocessor` 支持分割、姿态、OBB 等不同模型族。
- 后续可增加批量推理或视频帧输入，但首版先保证单图检测闭环。
