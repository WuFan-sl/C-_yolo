using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda.Preprocessing;

internal sealed record PreprocessingResult(
    DenseTensor<float> Tensor,
    LetterboxResult Letterbox);
