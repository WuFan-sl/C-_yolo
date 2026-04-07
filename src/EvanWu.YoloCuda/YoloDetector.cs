using EvanWu.YoloCuda.Postprocessing;
using EvanWu.YoloCuda.Preprocessing;
using EvanWu.YoloCuda.Runtime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda;

public sealed class YoloDetector : IDisposable
{
    private readonly YoloDetectorOptions _options;
    private readonly IYoloInferenceSession _session;
    private readonly IReadOnlyList<string> _labels;
    private bool _disposed;

    public YoloDetector(YoloDetectorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidateOptions(options);

        _options = options;
        _labels = LoadLabels(options.LabelsPath);
        _session = OnnxCudaSessionFactory.Create(options, out string mode, out int modelWidth, out int modelHeight);
        ExecutionMode = mode;

        if (modelWidth != options.InputWidth || modelHeight != options.InputHeight)
        {
            Console.WriteLine($"[Warning] Model expects {modelWidth}x{modelHeight}, options specify {options.InputWidth}x{options.InputHeight}. Using model dimensions.");
            _options = _options with { InputWidth = modelWidth, InputHeight = modelHeight };
        }

        try
        {
            ValidateModelShape(_session, _options);
        }
        catch
        {
            _session.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Indicates the execution device used: "CUDA" or "CPU".
    /// </summary>
    public string ExecutionMode { get; }

    /// <summary>
    /// Effective model input width used for preprocessing.
    /// </summary>
    public int InputWidth => _options.InputWidth;

    /// <summary>
    /// Effective model input height used for preprocessing.
    /// </summary>
    public int InputHeight => _options.InputHeight;

    internal YoloDetector(YoloDetectorOptions options, IYoloInferenceSession session, IReadOnlyList<string>? labels = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(session);
        ValidateOptions(options);

        _options = options;
        _labels = labels ?? LoadLabels(options.LabelsPath);
        _session = session;
        ExecutionMode = "Internal";

        ValidateModelShape(_session, options);
    }

    public IReadOnlyList<DetectionResult> Detect(string imagePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file was not found: {imagePath}", imagePath);
        }

        var preprocessor = new ImagePreprocessor();
        PreprocessingResult input = preprocessor.Preprocess(imagePath, _options.InputWidth, _options.InputHeight);

        Tensor<float> output = _session.Run(input.Tensor);
        var postprocessor = new YoloPostprocessor(_options.ConfidenceThreshold, _options.NmsThreshold, _labels);

        return postprocessor.Process(output, input.Letterbox);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _session.Dispose();
        _disposed = true;
    }

    private static void ValidateOptions(YoloDetectorOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.ModelPath))
        {
            throw new ArgumentException("Model path must be provided.", nameof(YoloDetectorOptions.ModelPath));
        }

        if (!File.Exists(options.ModelPath))
        {
            throw new FileNotFoundException($"Model file was not found: {options.ModelPath}", options.ModelPath);
        }

        if (!string.IsNullOrWhiteSpace(options.LabelsPath) && !File.Exists(options.LabelsPath))
        {
            throw new FileNotFoundException($"Labels file was not found: {options.LabelsPath}", options.LabelsPath);
        }

        if (options.InputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(YoloDetectorOptions.InputWidth));
        }

        if (options.InputHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(YoloDetectorOptions.InputHeight));
        }

        if (options.GpuDeviceId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(YoloDetectorOptions.GpuDeviceId));
        }

        ValidateThreshold(options.ConfidenceThreshold, nameof(YoloDetectorOptions.ConfidenceThreshold));
        ValidateThreshold(options.NmsThreshold, nameof(YoloDetectorOptions.NmsThreshold));
    }

    private static void ValidateThreshold(float value, string parameterName)
    {
        if (value is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static IReadOnlyList<string> LoadLabels(string? labelsPath)
    {
        if (string.IsNullOrWhiteSpace(labelsPath))
        {
            return Array.Empty<string>();
        }

        return File.ReadLines(labelsPath)
            .Select(label => label.Trim())
            .Where(label => label.Length > 0)
            .ToArray();
    }

    private static void ValidateModelShape(IYoloInferenceSession session, YoloDetectorOptions options)
    {
        if (session.InputDimensions.Count != 4)
        {
            throw new NotSupportedException($"Unsupported ONNX input shape: [{string.Join(", ", session.InputDimensions)}]");
        }

        if (!DimensionMatches(session.InputDimensions[0], 1)
            || !DimensionMatches(session.InputDimensions[1], 3)
            || !DimensionMatches(session.InputDimensions[2], options.InputHeight)
            || !DimensionMatches(session.InputDimensions[3], options.InputWidth))
        {
            throw new NotSupportedException($"Unsupported ONNX input shape: [{string.Join(", ", session.InputDimensions)}]");
        }

        if (session.OutputDimensions.Count != 3)
        {
            throw new NotSupportedException($"Unsupported ONNX output shape: [{string.Join(", ", session.OutputDimensions)}]");
        }
    }

    private static bool DimensionMatches(int modelDimension, int expected)
    {
        return modelDimension < 0 || modelDimension == expected;
    }
}
