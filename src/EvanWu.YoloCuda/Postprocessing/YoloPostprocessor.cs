using EvanWu.YoloCuda.Preprocessing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda.Postprocessing;

internal sealed class YoloPostprocessor
{
    private readonly float _confidenceThreshold;
    private readonly float _nmsThreshold;
    private readonly IReadOnlyList<string> _labels;
    private readonly string? _modelTask;

    public YoloPostprocessor(float confidenceThreshold, float nmsThreshold, IReadOnlyList<string>? labels, string? modelTask = null)
    {
        _confidenceThreshold = confidenceThreshold;
        _nmsThreshold = nmsThreshold;
        _labels = labels ?? Array.Empty<string>();
        _modelTask = modelTask;
    }

    public IReadOnlyList<DetectionResult> Process(Tensor<float> output, LetterboxResult letterbox)
    {
        ArgumentNullException.ThrowIfNull(output);

        int[] dimensions = output.Dimensions.ToArray();
        if (dimensions.Length != 3 || dimensions[0] != 1)
        {
            throw UnsupportedShape(dimensions);
        }

        var detections = new List<DetectionResult>();
        if (LooksLikeAttributeCount(dimensions[1], dimensions[2]))
        {
            ParseChannelsFirst(output, dimensions[1], dimensions[2], letterbox, detections);
        }
        else if (LooksLikeAttributeCount(dimensions[2], dimensions[1]))
        {
            ParseDetectionsFirst(output, dimensions[1], dimensions[2], letterbox, detections);
        }
        else
        {
            throw UnsupportedShape(dimensions);
        }

        return NonMaxSuppression.Select(detections, _nmsThreshold);
    }

    private void ParseChannelsFirst(
        Tensor<float> output,
        int attributeCount,
        int detectionCount,
        LetterboxResult letterbox,
        List<DetectionResult> detections)
    {
        if (attributeCount < 5)
        {
            throw UnsupportedShape(output.Dimensions.ToArray());
        }

        for (int detectionIndex = 0; detectionIndex < detectionCount; detectionIndex++)
        {
            float centerX = output[0, 0, detectionIndex];
            float centerY = output[0, 1, detectionIndex];
            float width = output[0, 2, detectionIndex];
            float height = output[0, 3, detectionIndex];
            AddDetection(output, letterbox, detections, detectionIndex, attributeCount, centerX, centerY, width, height, channelsFirst: true);
        }
    }

    private void ParseDetectionsFirst(
        Tensor<float> output,
        int detectionCount,
        int attributeCount,
        LetterboxResult letterbox,
        List<DetectionResult> detections)
    {
        if (attributeCount < 5)
        {
            throw UnsupportedShape(output.Dimensions.ToArray());
        }

        for (int detectionIndex = 0; detectionIndex < detectionCount; detectionIndex++)
        {
            float centerX = output[0, detectionIndex, 0];
            float centerY = output[0, detectionIndex, 1];
            float width = output[0, detectionIndex, 2];
            float height = output[0, detectionIndex, 3];
            AddDetection(output, letterbox, detections, detectionIndex, attributeCount, centerX, centerY, width, height, channelsFirst: false);
        }
    }

    private void AddDetection(
        Tensor<float> output,
        LetterboxResult letterbox,
        List<DetectionResult> detections,
        int detectionIndex,
        int attributeCount,
        float centerX,
        float centerY,
        float width,
        float height,
        bool channelsFirst)
    {
        bool isObb = IsObb(attributeCount);
        bool hasObjectness = !isObb && HasObjectness(attributeCount);
        int classStartIndex = hasObjectness ? 5 : 4;
        int classEndIndex = isObb ? attributeCount - 1 : attributeCount;
        float scoreMultiplier = hasObjectness
            ? ReadAttribute(output, detectionIndex, 4, channelsFirst)
            : 1f;

        int classId = -1;
        float confidence = float.MinValue;
        for (int attributeIndex = classStartIndex; attributeIndex < classEndIndex; attributeIndex++)
        {
            float classScore = ReadAttribute(output, detectionIndex, attributeIndex, channelsFirst);
            float score = scoreMultiplier * classScore;

            if (score > confidence)
            {
                classId = attributeIndex - classStartIndex;
                confidence = score;
            }
        }

        if (classId < 0 || confidence < _confidenceThreshold)
        {
            return;
        }

        float angle = isObb ? ReadAttribute(output, detectionIndex, attributeCount - 1, channelsFirst) : 0f;
        OrientedBoundingBox? orientedBox = isObb
            ? RestoreOrientedBox(centerX, centerY, width, height, angle, letterbox)
            : null;
        BoundingBox? restoredBox = orientedBox is null
            ? RestoreAndClipBox(centerX, centerY, width, height, letterbox)
            : RestoreAndClipRotatedBox(orientedBox.Value, letterbox);
        if (restoredBox is null)
        {
            return;
        }

        detections.Add(new DetectionResult(
            classId,
            GetLabel(classId),
            confidence,
            restoredBox.Value,
            orientedBox));
    }

    private bool HasObjectness(int attributeCount)
    {
        return _labels.Count > 0 && attributeCount == _labels.Count + 5;
    }

    private bool IsObb(int attributeCount)
    {
        return string.Equals(_modelTask, "obb", StringComparison.OrdinalIgnoreCase)
            && _labels.Count > 0
            && attributeCount == _labels.Count + 5;
    }

    private static float ReadAttribute(Tensor<float> output, int detectionIndex, int attributeIndex, bool channelsFirst)
    {
        return channelsFirst
            ? output[0, attributeIndex, detectionIndex]
            : output[0, detectionIndex, attributeIndex];
    }

    private bool LooksLikeAttributeCount(int candidateAttributeCount, int otherDimension)
    {
        if (candidateAttributeCount < 5)
        {
            return false;
        }

        int expectedWithoutObjectness = _labels.Count + 4;
        int expectedWithObjectness = _labels.Count + 5;
        if (_labels.Count > 0
            && (candidateAttributeCount == expectedWithoutObjectness || candidateAttributeCount == expectedWithObjectness))
        {
            return true;
        }

        return otherDimension < 5 || candidateAttributeCount <= otherDimension;
    }

    private string GetLabel(int classId)
    {
        return classId >= 0 && classId < _labels.Count
            ? _labels[classId]
            : $"class_{classId}";
    }

    private static BoundingBox? RestoreAndClipBox(
        float centerX,
        float centerY,
        float width,
        float height,
        LetterboxResult letterbox)
    {
        float left = (centerX - width / 2f - letterbox.PadX) / letterbox.Scale;
        float top = (centerY - height / 2f - letterbox.PadY) / letterbox.Scale;
        float right = (centerX + width / 2f - letterbox.PadX) / letterbox.Scale;
        float bottom = (centerY + height / 2f - letterbox.PadY) / letterbox.Scale;

        float clippedLeft = Math.Clamp(left, 0, letterbox.OriginalWidth);
        float clippedTop = Math.Clamp(top, 0, letterbox.OriginalHeight);
        float clippedRight = Math.Clamp(right, 0, letterbox.OriginalWidth);
        float clippedBottom = Math.Clamp(bottom, 0, letterbox.OriginalHeight);
        float clippedWidth = clippedRight - clippedLeft;
        float clippedHeight = clippedBottom - clippedTop;

        return clippedWidth <= 0 || clippedHeight <= 0
            ? null
            : new BoundingBox(clippedLeft, clippedTop, clippedWidth, clippedHeight);
    }

    private static OrientedBoundingBox RestoreOrientedBox(
        float centerX,
        float centerY,
        float width,
        float height,
        float angle,
        LetterboxResult letterbox)
    {
        return new OrientedBoundingBox(
            (centerX - letterbox.PadX) / letterbox.Scale,
            (centerY - letterbox.PadY) / letterbox.Scale,
            width / letterbox.Scale,
            height / letterbox.Scale,
            angle);
    }

    private static BoundingBox? RestoreAndClipRotatedBox(OrientedBoundingBox box, LetterboxResult letterbox)
    {
        float cos = Math.Abs(MathF.Cos(box.RotationRadians));
        float sin = Math.Abs(MathF.Sin(box.RotationRadians));
        float enclosingWidth = box.Width * cos + box.Height * sin;
        float enclosingHeight = box.Width * sin + box.Height * cos;

        float left = box.CenterX - enclosingWidth / 2f;
        float top = box.CenterY - enclosingHeight / 2f;
        float right = box.CenterX + enclosingWidth / 2f;
        float bottom = box.CenterY + enclosingHeight / 2f;

        float clippedLeft = Math.Clamp(left, 0, letterbox.OriginalWidth);
        float clippedTop = Math.Clamp(top, 0, letterbox.OriginalHeight);
        float clippedRight = Math.Clamp(right, 0, letterbox.OriginalWidth);
        float clippedBottom = Math.Clamp(bottom, 0, letterbox.OriginalHeight);
        float clippedWidth = clippedRight - clippedLeft;
        float clippedHeight = clippedBottom - clippedTop;

        return clippedWidth <= 0 || clippedHeight <= 0
            ? null
            : new BoundingBox(clippedLeft, clippedTop, clippedWidth, clippedHeight);
    }

    private static NotSupportedException UnsupportedShape(IReadOnlyList<int> dimensions)
    {
        return new NotSupportedException($"Unsupported YOLO output shape: [{string.Join(", ", dimensions)}]");
    }
}
