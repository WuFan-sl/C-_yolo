using Microsoft.ML.OnnxRuntime;

namespace EvanWu.YoloCuda.Runtime;

internal static class OnnxCudaSessionFactory
{
    public static IYoloInferenceSession Create(YoloDetectorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        try
        {
            using SessionOptions sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(options.GpuDeviceId);
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            var session = new InferenceSession(options.ModelPath, sessionOptions);
            return new OnnxCudaInferenceSession(session);
        }
        catch (OnnxRuntimeException ex)
        {
            throw new InvalidOperationException(
                $"Failed to initialize ONNX Runtime CUDA session for model '{options.ModelPath}' on GPU device {options.GpuDeviceId}: {ex.Message}",
                ex);
        }
    }
}
