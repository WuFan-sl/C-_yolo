# C# YOLO CUDA 类库需求文档

## 背景
项目需要一个可复用的 C# 类库，在 Windows + .NET 环境中加载 YOLO ONNX 模型，并通过 NVIDIA GPU 执行目标检测推理。
首版目标是把模型加载、图像预处理、CUDA 推理、YOLO 后处理和结果返回封装成稳定接口，供后续 Console、WinForms、WPF、服务端或工业视觉应用复用。

## 目标用户
- C# 应用开发者：需要在现有 Windows .NET 项目中集成本地目标检测能力。
- 产品工程师：需要稳定、可测试、可诊断的 GPU 推理组件，而不是一次性脚本。
- 部署工程师：需要明确的 CUDA/ONNX Runtime/.NET 依赖说明和故障定位入口。

## 首版目标
- 创建 `EvanWu.YoloCuda` 类库，提供简单的 YOLO 目标检测 API。
- 支持 YOLOv8/YOLO11 ONNX 检测模型。
- 使用 ONNX Runtime CUDA Execution Provider 在 NVIDIA GPU 上运行。
- 支持图片文件输入，返回类别、置信度和原图坐标系下的边界框。
- 提供 xUnit 单元测试，覆盖预处理、后处理和公共 API 行为。
- 提供可选 GPU 集成测试，用于 Windows NVIDIA 环境中的真实 ONNX 推理验证。

## 非目标
- 首版不支持 YOLO 分割、姿态估计、旋转框 OBB 或分类模型。
- 首版不实现训练、模型导出或标注工具。
- 首版不提交大型 `.onnx` 模型文件。
- 首版不提供 UI；UI 或桌面示例可在类库稳定后作为单独阶段实现。
- 首版不做 DirectML/WinML 双后端，除非后续明确扩展。

## 成功标准
- 外部项目能通过一个稳定入口调用检测，例如创建 `YoloDetector` 后对图片执行检测。
- 无 GPU/无模型环境下，单元测试仍可运行并验证核心算法。
- Windows NVIDIA 环境下，可通过显式 GPU 集成测试确认 CUDA provider 可用。
- 依赖、运行前置条件、模型放置方式和测试命令在文档中明确。

## 关键风险
- ONNX Runtime GPU 包与本机 CUDA/cuDNN 运行时不匹配会导致会话初始化失败。
- 不同 YOLO 导出版本的输出张量形状可能不同，后处理必须做输入/输出形状校验。
- 大模型文件不适合直接提交到普通 Git 仓库，需要明确模型资产管理策略。
