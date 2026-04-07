namespace EvanWu.YoloCuda;

public sealed record YoloDetectorOptions
{
    public required string ModelPath { get; init; }

    public string? LabelsPath { get; init; }

    public int InputWidth { get; init; } = 640;

    public int InputHeight { get; init; } = 640;

    public int GpuDeviceId { get; init; } = 0;

    public float ConfidenceThreshold { get; init; } = 0.25f;

    public float NmsThreshold { get; init; } = 0.45f;
}
