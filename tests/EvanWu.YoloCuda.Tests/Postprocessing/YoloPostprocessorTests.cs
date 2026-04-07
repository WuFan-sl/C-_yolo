using EvanWu.YoloCuda.Postprocessing;
using EvanWu.YoloCuda.Preprocessing;
using FluentAssertions;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda.Tests.Postprocessing;

public sealed class YoloPostprocessorTests
{
    [Fact]
    public void ProcessParsesYoloChannelsFirstOutputAndRestoresOriginalCoordinates()
    {
        var output = new DenseTensor<float>(new[] { 1, 6, 3 });
        SetChannelsFirstDetection(output, detectionIndex: 0, centerX: 320, centerY: 320, width: 64, height: 64, class0: 0.90f, class1: 0.10f);
        SetChannelsFirstDetection(output, detectionIndex: 1, centerX: 322, centerY: 322, width: 64, height: 64, class0: 0.80f, class1: 0.05f);
        SetChannelsFirstDetection(output, detectionIndex: 2, centerX: 96, centerY: 192, width: 32, height: 32, class0: 0.10f, class1: 0.70f);

        var postprocessor = new YoloPostprocessor(0.25f, 0.45f, new[] { "person", "bicycle" });
        var letterbox = new LetterboxResult(200, 100, 640, 640, 3.2f, 0f, 160f);

        IReadOnlyList<DetectionResult> results = postprocessor.Process(output, letterbox);

        results.Should().HaveCount(2);
        results[0].Label.Should().Be("person");
        results[0].Confidence.Should().BeApproximately(0.90f, 0.001f);
        results[0].Box.X.Should().BeApproximately(90f, 0.001f);
        results[0].Box.Y.Should().BeApproximately(40f, 0.001f);
        results[0].Box.Width.Should().BeApproximately(20f, 0.001f);
        results[0].Box.Height.Should().BeApproximately(20f, 0.001f);
        results[1].Label.Should().Be("bicycle");
    }

    [Fact]
    public void ProcessParsesYoloDetectionsFirstOutputAndFallsBackToClassLabel()
    {
        var output = new DenseTensor<float>(new[] { 1, 1, 6 });
        output[0, 0, 0] = 50;
        output[0, 0, 1] = 40;
        output[0, 0, 2] = 20;
        output[0, 0, 3] = 10;
        output[0, 0, 4] = 0.10f;
        output[0, 0, 5] = 0.85f;

        var postprocessor = new YoloPostprocessor(0.25f, 0.45f, labels: null);
        var letterbox = new LetterboxResult(100, 80, 100, 80, 1f, 0f, 0f);

        IReadOnlyList<DetectionResult> results = postprocessor.Process(output, letterbox);

        results.Should().ContainSingle();
        results[0].ClassId.Should().Be(1);
        results[0].Label.Should().Be("class_1");
        results[0].Box.Should().Be(new BoundingBox(40, 35, 20, 10));
    }

    [Fact]
    public void ProcessRejectsUnsupportedOutputShape()
    {
        var output = new DenseTensor<float>(new[] { 1, 4, 1 });
        var postprocessor = new YoloPostprocessor(0.25f, 0.45f, labels: null);
        var letterbox = new LetterboxResult(100, 80, 100, 80, 1f, 0f, 0f);

        Action act = () => postprocessor.Process(output, letterbox);

        act.Should().Throw<NotSupportedException>().WithMessage("*[1, 4, 1]*");
    }

    private static void SetChannelsFirstDetection(
        DenseTensor<float> output,
        int detectionIndex,
        float centerX,
        float centerY,
        float width,
        float height,
        float class0,
        float class1)
    {
        output[0, 0, detectionIndex] = centerX;
        output[0, 1, detectionIndex] = centerY;
        output[0, 2, detectionIndex] = width;
        output[0, 3, detectionIndex] = height;
        output[0, 4, detectionIndex] = class0;
        output[0, 5, detectionIndex] = class1;
    }
}
