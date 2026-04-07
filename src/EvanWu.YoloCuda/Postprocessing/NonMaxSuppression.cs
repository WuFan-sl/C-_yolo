namespace EvanWu.YoloCuda.Postprocessing;

internal static class NonMaxSuppression
{
    public static IReadOnlyList<DetectionResult> Select(IEnumerable<DetectionResult> detections, float iouThreshold)
    {
        ArgumentNullException.ThrowIfNull(detections);

        var selected = new List<DetectionResult>();
        foreach (IGrouping<int, DetectionResult> classGroup in detections.GroupBy(detection => detection.ClassId))
        {
            var candidates = classGroup.OrderByDescending(detection => detection.Confidence).ToList();
            while (candidates.Count > 0)
            {
                DetectionResult best = candidates[0];
                selected.Add(best);
                candidates.RemoveAt(0);
                candidates.RemoveAll(candidate => CalculateIou(best.Box, candidate.Box) > iouThreshold);
            }
        }

        return selected
            .OrderByDescending(detection => detection.Confidence)
            .ToList();
    }

    private static float CalculateIou(BoundingBox first, BoundingBox second)
    {
        float intersectionLeft = Math.Max(first.X, second.X);
        float intersectionTop = Math.Max(first.Y, second.Y);
        float intersectionRight = Math.Min(first.X + first.Width, second.X + second.Width);
        float intersectionBottom = Math.Min(first.Y + first.Height, second.Y + second.Height);

        float intersectionWidth = Math.Max(0, intersectionRight - intersectionLeft);
        float intersectionHeight = Math.Max(0, intersectionBottom - intersectionTop);
        float intersectionArea = intersectionWidth * intersectionHeight;
        float firstArea = Math.Max(0, first.Width) * Math.Max(0, first.Height);
        float secondArea = Math.Max(0, second.Width) * Math.Max(0, second.Height);
        float unionArea = firstArea + secondArea - intersectionArea;

        return unionArea <= 0 ? 0 : intersectionArea / unionArea;
    }
}
