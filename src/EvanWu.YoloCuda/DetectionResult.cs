namespace EvanWu.YoloCuda;

public sealed record DetectionResult(
    int ClassId,
    string Label,
    float Confidence,
    BoundingBox Box);
