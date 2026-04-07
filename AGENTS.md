# C# YOLO CUDA Repository Guidelines

## Project Overview & Core Concepts
- **Project**: C# YOLO CUDA class library for Windows + .NET.
- **Core Purpose**: Provide a reusable .NET class library that runs YOLO object detection models on NVIDIA GPUs through ONNX Runtime CUDA.
- **Primary Runtime**: Windows, C#, .NET 10 LTS, NVIDIA GPU, ONNX model format.
- **Model Target**: YOLOv8/YOLO11 ONNX object detection models, with limited YOLO OBB support when ONNX metadata marks `task=obb`. Segmentation and pose are out of scope for the first implementation.
- **Current State**: Initial `EvanWu.YoloCuda` solution, class library, and xUnit test project have been scaffolded and implemented for the first detection-library baseline.

## Project Structure & Module Organization
- Planned root layout:
  - `EvanWu.YoloCuda.sln`: solution file.
  - `src/EvanWu.YoloCuda/`: reusable class library.
  - `tests/EvanWu.YoloCuda.Tests/`: xUnit tests for preprocessing, postprocessing, API contracts, and optional GPU integration.
  - `docs/`: product, architecture, API, data model, technology, test strategy, and progress documents.
  - `samples/`: optional future console or desktop sample app after the core library is stable.
- Do not commit large model binaries by default. Keep `.onnx` files in a local `models/` directory or an external artifact store unless the user explicitly requests versioning.

## Current Focus & Quick Start
> This section gives agents immediate context without requiring a full read of every document.

- **Current Main Task**: Verify and refine the Windows C# .NET YOLO CUDA class library implementation, including ONNX metadata-aware detection/OBB postprocessing.
- **Primary Docs**:
  - Requirements: `docs/PRODUCT.md`
  - Architecture: `docs/ARCHITECTURE.md`
  - Public interface: `docs/API_CONTRACT.md`
  - Data model: `docs/DATA_MODEL.md`
  - Tech stack: `docs/TECH_STACK.md`
  - Test strategy: `docs/TEST_STRATEGY.md`
  - Progress: `docs/PROGRESS.md`
- **Next Implementation Step**: Validate real CUDA inference on a Windows NVIDIA machine with a local YOLO ONNX model.

## Build, Test, and Development Commands
- Check .NET SDK: `dotnet --info`
- Create solution: `dotnet new sln -n EvanWu.YoloCuda --format sln`
- Create class library: `dotnet new classlib -n EvanWu.YoloCuda -o src/EvanWu.YoloCuda -f net10.0`
- Create tests: `dotnet new xunit -n EvanWu.YoloCuda.Tests -o tests/EvanWu.YoloCuda.Tests -f net10.0`
- After project creation, set both project files to `TargetFramework` = `net10.0-windows`; set `EnableWindowsTargeting=true` when building on non-Windows development hosts.
- Add projects to solution:
  - `dotnet sln add src/EvanWu.YoloCuda/EvanWu.YoloCuda.csproj`
  - `dotnet sln add tests/EvanWu.YoloCuda.Tests/EvanWu.YoloCuda.Tests.csproj`
- Add test project reference: `dotnet add tests/EvanWu.YoloCuda.Tests/EvanWu.YoloCuda.Tests.csproj reference src/EvanWu.YoloCuda/EvanWu.YoloCuda.csproj`
- Runtime packages:
  - `Microsoft.ML.OnnxRuntime.Gpu.Windows` `1.24.4`
  - `SixLabors.ImageSharp` `3.1.12`
- Test packages:
  - `FluentAssertions` `8.9.0`
  - `xunit` `2.9.3`
- Run all tests: `dotnet test`
- Run optional GPU integration tests only when configured: `EVANWU_YOLO_ENABLE_GPU_TESTS=1 EVANWU_YOLO_TEST_MODEL=/path/to/model.onnx dotnet test --filter Category=Gpu`

## Coding Style & Naming Conventions
- Use modern C# with nullable reference types enabled and `ImplicitUsings` enabled.
- Namespaces should start with `EvanWu.YoloCuda`.
- Public types use `PascalCase`; methods use `PascalCase`; private fields use `_camelCase`; locals and parameters use `camelCase`.
- Public APIs should be small and stable. Prefer immutable records for result/value objects.
- `YoloDetector` must implement `IDisposable` or `IAsyncDisposable` when holding ONNX Runtime sessions or unmanaged resources.
- Do not silently fall back from CUDA to CPU in production paths. If GPU initialization fails, return a clear diagnostic error unless a future option explicitly enables fallback.

## Testing Guidelines
- Use xUnit for unit and integration tests.
- Unit tests must not require a real ONNX model or GPU. Test preprocessing, postprocessing, thresholds, NMS, and coordinate restoration with deterministic in-memory data.
- GPU tests must be opt-in via trait/category and require Windows + NVIDIA driver + compatible CUDA/cuDNN runtime dependencies.
- Keep test assets small. Avoid committing large images or `.onnx` model files unless explicitly approved.
- Before claiming implementation completion, run the relevant verification command and report the exact command and outcome.

## Documentation Index
- Product requirements: `docs/PRODUCT.md`
- Architecture: `docs/ARCHITECTURE.md`
- API contract: `docs/API_CONTRACT.md`
- Data model: `docs/DATA_MODEL.md`
- Technology stack: `docs/TECH_STACK.md`
- Test strategy: `docs/TEST_STRATEGY.md`
- Progress log: `docs/PROGRESS.md`

## Security & Configuration Tips
- Do not commit credentials, commercial model weights, customer images, or GPU runtime DLLs without explicit approval.
- Keep machine-specific paths configurable through options, environment variables, or sample configuration files.
- Validate image paths, model paths, label files, image dimensions, and model tensor shapes before inference.
- Surface dependency/version errors clearly because CUDA provider failures are often deployment-environment issues.

## Agent Workflow Hooks

This project enforces documentation, verification, and commit hygiene.

### Hook 1: Post-Code-Change Documentation Sync
After completing any code modification, debugging, or feature implementation task, analyze whether the change affects:
- Project structure or dependencies.
- Public API contracts or data models.
- Current development progress or known issues.
- Development workflow or commands.

Required documentation updates:
- Update `AGENTS.md` if commands, current focus, or structure changed.
- Always update `docs/PROGRESS.md`.
- Update `docs/API_CONTRACT.md` if public APIs changed.
- Update `docs/DATA_MODEL.md` if public data structures changed.
- Update `docs/TECH_STACK.md` if dependencies or runtime requirements changed.

### Hook 2: Post-Completion Git Commit
When the project is inside a Git repository and a feature/modification task is complete:
- Verify the change first.
- Stage functional and documentation changes together.
- Create an atomic Conventional Commit such as `feat(yolo): add cuda detector`.

If the directory is not a Git repository, do not initialize Git unless the user asks. Report that the commit step was skipped because no repository exists.

### Hook 3: Frontend Verification
There is no frontend in the current planned scope. If a future `frontend/` is added, browser verification requirements must be added before completing frontend work.

### Hook 4: Regression Gate
For the current class library scope, regression coverage means xUnit tests plus opt-in GPU integration tests. If a future frontend is added, add Playwright requirements before claiming frontend behavior is complete.
