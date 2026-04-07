using Microsoft.ML.OnnxRuntime;

namespace EvanWu.YoloCuda.Runtime;

internal static class OnnxCudaSessionFactory
{
    public static IYoloInferenceSession Create(YoloDetectorOptions options, out string executionMode, out int modelWidth, out int modelHeight)
    {
        ArgumentNullException.ThrowIfNull(options);
        modelWidth = 640;
        modelHeight = 640;

        try
        {
            using SessionOptions sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(options.GpuDeviceId);
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            var session = new InferenceSession(options.ModelPath, sessionOptions);
            (modelWidth, modelHeight) = ExtractInputSize(session.InputMetadata);
            executionMode = "CUDA";
            return new OnnxCudaInferenceSession(session);
        }
        catch (Exception)
        {
            using SessionOptions cpuOptions = new SessionOptions();
            cpuOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            var cpuSession = new InferenceSession(options.ModelPath, cpuOptions);
            (modelWidth, modelHeight) = ExtractInputSize(cpuSession.InputMetadata);
            executionMode = "CPU";
            return new OnnxCudaInferenceSession(cpuSession);
        }
    }

    private static (int width, int height) ExtractInputSize(IReadOnlyDictionary<string, NodeMetadata> metadata)
    {
        var inputMeta = metadata.Values.FirstOrDefault(m => m.Dimensions.Length == 4);
        if (inputMeta == null)
        {
            return (640, 640);
        }

        var dims = inputMeta.Dimensions;
        return (dims[3] < 0 ? 640 : dims[3], dims[2] < 0 ? 640 : dims[2]);
    }
}
