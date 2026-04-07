using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

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
        ModelTask = TryGetMetadataValue(session, "task");
        ModelLabels = ParseModelLabels(TryGetMetadataValue(session, "names"));
    }

    public string InputName { get; }

    public string OutputName { get; }

    public IReadOnlyList<int> InputDimensions { get; }

    public IReadOnlyList<int> OutputDimensions { get; }

    public string? ModelTask { get; }

    public IReadOnlyList<string> ModelLabels { get; }

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

    private static string? TryGetMetadataValue(InferenceSession session, string key)
    {
        return session.ModelMetadata.CustomMetadataMap.TryGetValue(key, out string? value) && !string.IsNullOrWhiteSpace(value)
            ? value
            : null;
    }

    private static IReadOnlyList<string> ParseModelLabels(string? names)
    {
        if (string.IsNullOrWhiteSpace(names))
        {
            return Array.Empty<string>();
        }

        var labelsById = new SortedDictionary<int, string>();
        foreach (Match match in Regex.Matches(names, @"(\d+)\s*:\s*['""]([^'""]+)['""]"))
        {
            labelsById[int.Parse(match.Groups[1].Value)] = match.Groups[2].Value;
        }

        return labelsById.Count == 0
            ? Array.Empty<string>()
            : labelsById.Values.ToArray();
    }
}
