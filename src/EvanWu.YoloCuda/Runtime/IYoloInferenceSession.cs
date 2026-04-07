using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda.Runtime;

internal interface IYoloInferenceSession : IDisposable
{
    string InputName { get; }

    string OutputName { get; }

    IReadOnlyList<int> InputDimensions { get; }

    IReadOnlyList<int> OutputDimensions { get; }

    Tensor<float> Run(DenseTensor<float> inputTensor);
}
