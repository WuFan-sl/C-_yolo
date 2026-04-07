# C# YOLO CUDA 技术栈文档

## 运行环境
- 操作系统：Windows 10/11 x64。
- 开发框架：.NET 10 LTS，目标框架 `net10.0-windows`。
- 编程语言：C#。
- GPU：NVIDIA GPU。
- 推理模型：YOLOv8/YOLO11 ONNX detection model。

## 核心依赖
- `Microsoft.ML.OnnxRuntime.Gpu`：ONNX Runtime C# 绑定和 CUDA Execution Provider。
- `SixLabors.ImageSharp`：图片读取、像素访问、缩放和预处理。
- `xunit`：测试框架。
- `FluentAssertions`：测试断言增强。

## 版本基线
- .NET SDK：`10.0.201` 已用于当前 Linux 开发环境验证。
- Target framework：`net10.0-windows`。
- `Microsoft.ML.OnnxRuntime.Gpu`：`1.24.4`。
- `SixLabors.ImageSharp`：`3.1.12`。
- `FluentAssertions`：`8.9.0`。
- `xunit`：`2.9.3`。

## CUDA 依赖说明
ONNX Runtime GPU 对 CUDA/cuDNN 运行时版本敏感。部署前必须核对：
- NVIDIA 驱动可用。
- CUDA provider 需要的 CUDA/cuDNN 运行时 DLL 可被进程加载。
- `GpuDeviceId` 与目标机器的 GPU 编号一致。

首版不把 CUDA/cuDNN DLL 纳入仓库。部署文档应说明从系统安装、ONNX Runtime 推荐方式或应用安装包中提供这些依赖。

## 开发命令基线
```bash
dotnet --info
dotnet test
EVANWU_YOLO_ENABLE_GPU_TESTS=1 EVANWU_YOLO_TEST_MODEL=/path/to/model.onnx dotnet test --filter Category=Gpu
```

当前项目已创建 `EvanWu.YoloCuda.sln`、`src/EvanWu.YoloCuda/EvanWu.YoloCuda.csproj` 和 `tests/EvanWu.YoloCuda.Tests/EvanWu.YoloCuda.Tests.csproj`。
在非 Windows 主机上构建 `net10.0-windows` 时，项目文件保留 `EnableWindowsTargeting=true`。

## 参考资料
- .NET support policy: https://dotnet.microsoft.com/en-us/platform/support/policy/dotnet-core
- ONNX Runtime C# docs: https://onnxruntime.ai/docs/get-started/with-csharp.html
- ONNX Runtime CUDA provider docs: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime GPU NuGet: https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/
- ImageSharp NuGet: https://www.nuget.org/packages/SixLabors.ImageSharp/
