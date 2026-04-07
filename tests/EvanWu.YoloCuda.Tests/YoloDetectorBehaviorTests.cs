using EvanWu.YoloCuda.Runtime;
using FluentAssertions;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda.Tests;

public sealed class YoloDetectorBehaviorTests
{
    [Fact]
    public void DetectRejectsMissingImagePathBeforeRunningInference()
    {
        using TestModelFile modelFile = TestModelFile.Create();
        using var detector = new YoloDetector(
            new YoloDetectorOptions { ModelPath = modelFile.Path },
            new FakeInferenceSession());
        string imagePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");

        Action act = () => detector.Detect(imagePath);

        act.Should().Throw<FileNotFoundException>().WithMessage($"*{imagePath}*");
    }

    [Fact]
    public void DetectAfterDisposeThrowsObjectDisposedException()
    {
        using TestModelFile modelFile = TestModelFile.Create();
        var detector = new YoloDetector(
            new YoloDetectorOptions { ModelPath = modelFile.Path },
            new FakeInferenceSession());

        detector.Dispose();

        Action act = () => detector.Detect("image.png");

        act.Should().Throw<ObjectDisposedException>().WithMessage("*YoloDetector*");
    }

    private sealed class FakeInferenceSession : IYoloInferenceSession
    {
        public string InputName => "images";

        public string OutputName => "output0";

        public IReadOnlyList<int> InputDimensions => new[] { 1, 3, 640, 640 };

        public IReadOnlyList<int> OutputDimensions => new[] { 1, 84, 8400 };

        public string? ModelTask => null;

        public IReadOnlyList<string> ModelLabels => Array.Empty<string>();

        public Tensor<float> Run(DenseTensor<float> inputTensor)
        {
            return new DenseTensor<float>(new[] { 1, 84, 0 });
        }

        public void Dispose()
        {
        }
    }

    private sealed class TestModelFile : IDisposable
    {
        private TestModelFile(string path)
        {
            Path = path;
        }

        public string Path { get; }

        public static TestModelFile Create()
        {
            return new TestModelFile(System.IO.Path.GetTempFileName());
        }

        public void Dispose()
        {
            File.Delete(Path);
        }
    }
}
