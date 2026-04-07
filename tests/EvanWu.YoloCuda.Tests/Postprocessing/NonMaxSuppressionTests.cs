using EvanWu.YoloCuda.Postprocessing;
using FluentAssertions;

namespace EvanWu.YoloCuda.Tests.Postprocessing;

public sealed class NonMaxSuppressionTests
{
    [Fact]
    public void SelectKeepsHighestConfidenceBoxForSameClassOverlap()
    {
        var detections = new[]
        {
            new DetectionResult(0, "person", 0.90f, new BoundingBox(10, 10, 20, 20)),
            new DetectionResult(0, "person", 0.80f, new BoundingBox(12, 12, 20, 20)),
            new DetectionResult(1, "car", 0.70f, new BoundingBox(12, 12, 20, 20))
        };

        IReadOnlyList<DetectionResult> selected = NonMaxSuppression.Select(detections, 0.45f);

        selected.Should().HaveCount(2);
        selected.Should().Contain(detections[0]);
        selected.Should().Contain(detections[2]);
        selected.Should().NotContain(detections[1]);
    }
}
