using FluentAssertions;

namespace EvanWu.YoloCuda.Tests.Runtime;

public sealed class GpuIntegrationTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void DetectorCanBeConstructedWhenGpuModelIsConfigured()
    {
        if (Environment.GetEnvironmentVariable("EVANWU_YOLO_ENABLE_GPU_TESTS") != "1")
        {
            return;
        }

        string? modelPath = Environment.GetEnvironmentVariable("EVANWU_YOLO_TEST_MODEL");
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            return;
        }

        using var detector = new YoloDetector(new YoloDetectorOptions { ModelPath = modelPath });

        detector.Should().NotBeNull();
    }
}
