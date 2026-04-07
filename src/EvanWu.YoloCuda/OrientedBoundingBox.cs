namespace EvanWu.YoloCuda;

public readonly record struct OrientedBoundingBox(
    float CenterX,
    float CenterY,
    float Width,
    float Height,
    float RotationRadians);
