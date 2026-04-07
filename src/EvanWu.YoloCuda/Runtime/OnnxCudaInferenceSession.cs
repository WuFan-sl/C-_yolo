using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EvanWu.YoloCuda.Runtime;

internal sealed class OnnxCudaInferenceSession : IYoloInferenceSession
{
    private readonly InferenceSession _session;

    public OnnxCudaInferenceSession(InferenceSession session)
    {
        _session = session;
        InputName = session.InputMetadata.Keys.Single();
        OutputName = session.OutputMetadata.Keys.Single();
        InputDimensions = session.InputMetadata[InputName].Dimensions.ToArray();
        OutputDimensions = session.OutputMetadata[OutputName].Dimensions.ToArray();
    }

    public string InputName { get; }

    public string OutputName { get; }

    public IReadOnlyList<int> InputDimensions { get; }

    public IReadOnlyList<int> OutputDimensions { get; }

    public Tensor<float> Run(DenseTensor<float> inputTensor)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, inputTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        Tensor<float> output = results.Single(result => result.Name == OutputName).AsTensor<float>();

        return new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
