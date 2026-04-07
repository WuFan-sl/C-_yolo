using FluentAssertions;

namespace EvanWu.YoloCuda.Tests;

public sealed class YoloDetectorOptionsTests
{
    [Fact]
    public void ConstructorRejectsNullOptions()
    {
        Action act = () => _ = new YoloDetector(null!);

        act.Should().Throw<ArgumentNullException>();
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void ConstructorRejectsEmptyModelPath(string modelPath)
    {
        Action act = () => _ = new YoloDetector(new YoloDetectorOptions { ModelPath = modelPath });

        act.Should().Throw<ArgumentException>().WithParameterName("ModelPath");
    }

    [Fact]
    public void ConstructorRejectsMissingModelPath()
    {
        string modelPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".onnx");

        Action act = () => _ = new YoloDetector(new YoloDetectorOptions { ModelPath = modelPath });

        act.Should().Throw<FileNotFoundException>().WithMessage($"*{modelPath}*");
    }

    [Fact]
    public void ConstructorRejectsMissingLabelsPath()
    {
        string modelPath = Path.GetTempFileName();
        string labelsPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".names");

        try
        {
            Action act = () => _ = new YoloDetector(new YoloDetectorOptions
            {
                ModelPath = modelPath,
                LabelsPath = labelsPath
            });

            act.Should().Throw<FileNotFoundException>().WithMessage($"*{labelsPath}*");
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Theory]
    [InlineData(0, 640, nameof(YoloDetectorOptions.InputWidth))]
    [InlineData(640, 0, nameof(YoloDetectorOptions.InputHeight))]
    public void ConstructorRejectsInvalidInputSize(int width, int height, string parameterName)
    {
        string modelPath = Path.GetTempFileName();

        try
        {
            Action act = () => _ = new YoloDetector(new YoloDetectorOptions
            {
                ModelPath = modelPath,
                InputWidth = width,
                InputHeight = height
            });

            act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameterName);
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Theory]
    [InlineData(-0.01f, 0.45f, nameof(YoloDetectorOptions.ConfidenceThreshold))]
    [InlineData(1.01f, 0.45f, nameof(YoloDetectorOptions.ConfidenceThreshold))]
    [InlineData(0.25f, -0.01f, nameof(YoloDetectorOptions.NmsThreshold))]
    [InlineData(0.25f, 1.01f, nameof(YoloDetectorOptions.NmsThreshold))]
    public void ConstructorRejectsInvalidThresholds(float confidenceThreshold, float nmsThreshold, string parameterName)
    {
        string modelPath = Path.GetTempFileName();

        try
        {
            Action act = () => _ = new YoloDetector(new YoloDetectorOptions
            {
                ModelPath = modelPath,
                ConfidenceThreshold = confidenceThreshold,
                NmsThreshold = nmsThreshold
            });

            act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameterName);
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact]
    public void ConstructorRejectsInvalidGpuDeviceId()
    {
        string modelPath = Path.GetTempFileName();

        try
        {
            Action act = () => _ = new YoloDetector(new YoloDetectorOptions
            {
                ModelPath = modelPath,
                GpuDeviceId = -1
            });

            act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(YoloDetectorOptions.GpuDeviceId));
        }
        finally
        {
            File.Delete(modelPath);
        }
    }
}
