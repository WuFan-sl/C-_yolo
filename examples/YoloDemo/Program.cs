using EvanWu.YoloCuda;
using System.Drawing;

const string ImgsDir = "./imgs";
const string OutputDir = "./output";
const string ModelsDir = "./models";

Console.WriteLine("=== YOLO CUDA DEMO ===");

if (!Directory.Exists(ImgsDir))
{
    Console.WriteLine($"Error: '{ImgsDir}' directory not found.");
    Console.WriteLine("Please create the 'imgs' folder and place your images there.");
    WaitForExit();
    return;
}

if (!Directory.Exists(ModelsDir))
{
    Console.WriteLine($"Error: '{ModelsDir}' directory not found.");
    Console.WriteLine("Please create the 'models' folder and place your .onnx model file there.");
    WaitForExit();
    return;
}

var modelResult = FindAndPrepareModel(ModelsDir);
if (modelResult == null)
{
    Console.WriteLine("No valid .onnx model file found in './models'.");
    Console.WriteLine("Please place a YOLO .onnx model file in the 'models' folder.");
    Console.WriteLine("For .pt models, use the provided convert.py script to convert to ONNX first.");
    WaitForExit();
    return;
}

var (modelPath, labelsPath) = modelResult.Value;

var options = new YoloDetectorOptions
{
    ModelPath = modelPath,
    LabelsPath = labelsPath,
    InputWidth = 640,
    InputHeight = 640,
    ConfidenceThreshold = 0.25f,
    NmsThreshold = 0.45f
};

Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
Console.WriteLine($"Labels: {labelsPath ?? "(none)"}");
Console.WriteLine($"Requested Input Size: {options.InputWidth}x{options.InputHeight}");
Console.WriteLine($"Confidence Threshold: {options.ConfidenceThreshold}");
Console.WriteLine($"NMS Threshold: {options.NmsThreshold}");
Console.WriteLine();

var imageFiles = Directory.GetFiles(ImgsDir, "*.*")
    .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
             || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
             || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
             || f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
    .OrderBy(f => f)
    .ToArray();

if (imageFiles.Length == 0)
{
    Console.WriteLine($"No images found in '{ImgsDir}'.");
    Console.WriteLine("Please add image files (.jpg, .jpeg, .png, .bmp) to the 'imgs' folder.");
    WaitForExit();
    return;
}

Console.WriteLine($"Found {imageFiles.Length} image(s) in '{ImgsDir}'");
Console.WriteLine();

Directory.CreateDirectory(OutputDir);

using var detector = new YoloDetector(options);

Console.WriteLine($"Execution: {detector.ExecutionMode}");
Console.WriteLine($"Actual Input Size: {detector.InputWidth}x{detector.InputHeight}");
Console.WriteLine();

foreach (var imagePath in imageFiles)
{
    var fileName = Path.GetFileName(imagePath);
    var outputFileName = Path.GetFileNameWithoutExtension(imagePath) + "_result.png";
    var outputPath = Path.Combine(OutputDir, outputFileName);

    Console.WriteLine($"--- Processing: {fileName} ---");

    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
    var results = detector.Detect(imagePath);
    stopwatch.Stop();

    Console.WriteLine($"Inference Time: {stopwatch.ElapsedMilliseconds} ms");
    Console.WriteLine($"Detections: {results.Count}");

    if (results.Count == 0)
    {
        Console.WriteLine("No objects detected.");

        using (var img = Image.FromFile(imagePath))
        using (var g = Graphics.FromImage(img))
        {
            g.DrawString("No Detection", new Font("Arial", 24), Brushes.Red, new PointF(10, 10));
            img.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
        }
    }
    else
    {
        foreach (var result in results)
        {
            Console.WriteLine($"  Class: {result.Label} (ID: {result.ClassId})");
            Console.WriteLine($"  Confidence: {result.Confidence:P2}");
            Console.WriteLine($"  Box: X={result.Box.X:F1}, Y={result.Box.Y:F1}, W={result.Box.Width:F1}, H={result.Box.Height:F1}");
        }

        using (var img = Image.FromFile(imagePath))
        using (var g = Graphics.FromImage(img))
        using (var pen = new Pen(Color.Lime, 2))
        using (var font = new Font("Arial", 12))
        {
            foreach (var result in results)
            {
                var rect = new RectangleF(result.Box.X, result.Box.Y, result.Box.Width, result.Box.Height);
                g.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);

                var label = $"{result.Label} {result.Confidence:P0}";
                var size = g.MeasureString(label, font);

                g.FillRectangle(new SolidBrush(Color.FromArgb(180, 0, 0, 0)), rect.X, rect.Y - size.Height, size.Width, size.Height);
                g.DrawString(label, font, Brushes.Lime, rect.X, rect.Y - size.Height);
            }

            img.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
        }

        Console.WriteLine($"Output saved: {outputPath}");
    }

    Console.WriteLine();
}

Console.WriteLine("=== DEMO Complete ===");
Console.WriteLine();
WaitForExit();

static void WaitForExit()
{
    Console.WriteLine("Press any key to exit...");
    Console.ReadKey();
}

static (string modelPath, string? labelsPath)? FindAndPrepareModel(string modelsDir)
{
    var onnxFiles = Directory.GetFiles(modelsDir, "*.onnx")
        .Concat(Directory.GetFiles(modelsDir, "*.ONNX"))
        .Distinct()
        .ToArray();

    if (onnxFiles.Length == 0)
    {
        var ptFiles = Directory.GetFiles(modelsDir, "*.pt")
            .Concat(Directory.GetFiles(modelsDir, "*.PT"))
            .ToArray();

        if (ptFiles.Length > 0)
        {
            Console.WriteLine("Found .pt model file(s), but ONNX format is required.");
            Console.WriteLine("Please run 'python convert.py' to convert the model to ONNX format.");
            Console.WriteLine();
        }

        return null;
    }

    var onnxFile = onnxFiles.First();
    var labelsFile = FindLabelsFile(modelsDir);

    return (onnxFile, labelsFile);
}

static string? FindLabelsFile(string modelsDir)
{
    var possibleNames = new[] { "coco.names", "coco.txt", "labels.txt", "classes.txt", "names.txt" };

    foreach (var name in possibleNames)
    {
        var path = Path.Combine(modelsDir, name);
        if (File.Exists(path))
        {
            return path;
        }
    }

    var txtFiles = Directory.GetFiles(modelsDir, "*.txt");
    return txtFiles.FirstOrDefault();
}
