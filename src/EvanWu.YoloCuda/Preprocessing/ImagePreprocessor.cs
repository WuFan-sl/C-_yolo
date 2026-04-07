using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace EvanWu.YoloCuda.Preprocessing;

internal sealed class ImagePreprocessor
{
    public PreprocessingResult Preprocess(string imagePath, int inputWidth, int inputHeight)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file was not found: {imagePath}", imagePath);
        }

        if (inputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputWidth));
        }

        if (inputHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputHeight));
        }

        try
        {
            using Image<Rgb24> image = Image.Load<Rgb24>(imagePath);
            float scale = Math.Min(inputWidth / (float)image.Width, inputHeight / (float)image.Height);
            int resizedWidth = Math.Max(1, (int)Math.Round(image.Width * scale));
            int resizedHeight = Math.Max(1, (int)Math.Round(image.Height * scale));
            int padX = (inputWidth - resizedWidth) / 2;
            int padY = (inputHeight - resizedHeight) / 2;

            using Image<Rgb24> resized = image.Clone(context => context.Resize(
                new ResizeOptions
                {
                    Size = new Size(resizedWidth, resizedHeight),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Triangle
                }));

            using var canvas = new Image<Rgb24>(inputWidth, inputHeight, new Rgb24(114, 114, 114));
            CopyIntoCanvas(resized, canvas, padX, padY);

            var tensor = new DenseTensor<float>(new[] { 1, 3, inputHeight, inputWidth });
            for (int y = 0; y < inputHeight; y++)
            {
                for (int x = 0; x < inputWidth; x++)
                {
                    Rgb24 pixel = canvas[x, y];
                    tensor[0, 0, y, x] = pixel.R / 255f;
                    tensor[0, 1, y, x] = pixel.G / 255f;
                    tensor[0, 2, y, x] = pixel.B / 255f;
                }
            }

            var letterbox = new LetterboxResult(
                image.Width,
                image.Height,
                inputWidth,
                inputHeight,
                scale,
                padX,
                padY);

            return new PreprocessingResult(tensor, letterbox);
        }
        catch (UnknownImageFormatException ex)
        {
            throw new InvalidDataException($"Image file could not be decoded: {imagePath}", ex);
        }
        catch (InvalidImageContentException ex)
        {
            throw new InvalidDataException($"Image file could not be decoded: {imagePath}", ex);
        }
    }

    private static void CopyIntoCanvas(Image<Rgb24> source, Image<Rgb24> destination, int offsetX, int offsetY)
    {
        for (int y = 0; y < source.Height; y++)
        {
            for (int x = 0; x < source.Width; x++)
            {
                destination[x + offsetX, y + offsetY] = source[x, y];
            }
        }
    }
}
