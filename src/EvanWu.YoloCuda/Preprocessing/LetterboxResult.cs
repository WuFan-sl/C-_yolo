namespace EvanWu.YoloCuda.Preprocessing;

internal readonly record struct LetterboxResult(
    int OriginalWidth,
    int OriginalHeight,
    int InputWidth,
    int InputHeight,
    float Scale,
    float PadX,
    float PadY);
