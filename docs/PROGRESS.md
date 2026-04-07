# 开发进度

## 2026-04-07
- 初始化 C# YOLO CUDA 类库的文档基线。
- 明确首版产品方向：Windows + C# + .NET 10 LTS + ONNX Runtime CUDA + YOLOv8/YOLO11 ONNX 目标检测。
- 完善 `docs/API_CONTRACT.md`：补充 v1 范围、NuGet/运行依赖、公共类型字段契约、模型 I/O、错误处理、示例和测试契约映射。
- 对齐 `docs/DATA_MODEL.md` 中标签文件可选/缺失的措辞：未提供标签文件时回退，配置了不存在的标签文件路径时应抛错。
- 移除小写 `agents.md` 约定，项目仅使用 `AGENTS.md` 作为代理工作指南。
- 当前目录不是 Git 仓库，暂时无法执行提交钩子。
- 当前本机 shell 未检测到 `dotnet`，后续创建和测试 .NET 项目需要在安装 .NET 10 SDK 的 Windows 环境执行。
- 确认最终项目名和命名空间为 `EvanWu.YoloCuda`，并同步替换文档中的旧 `WuFan` 命名。
- 通过用户级安装脚本安装并使用 `.NET SDK 10.0.201` 到 `~/.dotnet/dotnet-sdk-10`，未使用 sudo 或系统包管理器。
- 创建 `EvanWu.YoloCuda.sln`、`src/EvanWu.YoloCuda` 类库和 `tests/EvanWu.YoloCuda.Tests` xUnit 测试项目。
- 添加并锁定依赖：`Microsoft.ML.OnnxRuntime.Gpu` `1.24.4`、`SixLabors.ImageSharp` `3.1.12`、`FluentAssertions` `8.9.0`、`xunit` `2.9.3`。
- 实现首版公共 API：`YoloDetectorOptions`、`YoloDetector`、`DetectionResult`、`BoundingBox`。
- 实现核心模块：图片 letterbox 预处理、RGB NCHW float tensor 生成、YOLO detection 输出解析、坐标还原、NMS、ONNX Runtime CUDA session factory。
- 添加 `IYoloInferenceSession` 适配层，允许单元测试注入 fake session，避免默认测试依赖真实 ONNX 模型或 GPU。
- 添加 `.gitignore`，忽略 `bin/`、`obj/`、本地 `models/` 和 `.onnx` 模型文件。
- TDD 记录：先运行 `dotnet test EvanWu.YoloCuda.sln --no-restore`，测试因缺少计划中的类型/模块编译失败；实现后测试通过。
- 验证命令：`dotnet build EvanWu.YoloCuda.sln --no-restore` 通过，0 warnings / 0 errors。
- 验证命令：`dotnet test EvanWu.YoloCuda.sln` 通过，结果为 21 passed / 0 failed / 0 skipped。
- 验证命令：`dotnet test EvanWu.YoloCuda.sln --filter Category=Gpu` 通过，结果为 1 passed / 0 failed / 0 skipped；当前未设置 `EVANWU_YOLO_ENABLE_GPU_TESTS=1` 和真实模型路径，因此只验证 GPU 测试门控，不代表真实 CUDA 推理已验证。
- 排查 `best.onnx` 转换后检测效果差的问题：确认示例模型输入为 `[1, 3, 320, 320]`、输出为 `[1, 6, 2100]`，并修复单类别带 objectness 输出被误当成两类别输出的问题。
- 预处理对齐 Ultralytics 常见 letterbox 行为：padding 改为 `114,114,114`，缩放采样改为 `KnownResamplers.Triangle`，降低 ONNX 推理输入分布偏差。
- 为 `YoloDetector` 添加 `InputWidth` / `InputHeight` 只读属性，demo 可显示实际使用的模型输入尺寸，避免固定打印 `640x640` 造成误判。
- 同步文档中的 ONNX Runtime GPU Windows 包名和 CPU fallback 诊断约束。
- 验证命令：`dotnet test` 通过，结果为 23 passed / 0 failed / 0 skipped。

## 当前状态
- 已完成需求、架构、接口、数据模型、技术栈、测试策略文档规划；接口文档已扩展为可指导首版实现的 API 契约。
- 已完成首版类库源码和 xUnit 测试项目；默认单元测试在当前 Linux 开发环境中通过。
- GPU 集成测试已加 `[Trait("Category", "Gpu")]`，并要求 `EVANWU_YOLO_ENABLE_GPU_TESTS=1` 和 `EVANWU_YOLO_TEST_MODEL` 才尝试真实 detector 初始化。

## 下一步
- 在 Windows + NVIDIA + CUDA/cuDNN 环境中准备本地 YOLOv8/YOLO11 detection ONNX 模型。
- 运行 `EVANWU_YOLO_ENABLE_GPU_TESTS=1 EVANWU_YOLO_TEST_MODEL=/path/to/model.onnx dotnet test --filter Category=Gpu` 验证真实 CUDA session 初始化。
- 后续可补充端到端样例图片与更完整的模型输出形状兼容性测试。
