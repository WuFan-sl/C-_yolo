using EvanWu.YoloCuda.Preprocessing;
using FluentAssertions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace EvanWu.YoloCuda.Tests.Preprocessing;

public sealed class ImagePreprocessorTests
{
    [Fact]
    public void PreprocessConvertsPixelsToRgbNchwFloatTensor()
    {
        string imagePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");

        try
        {
            using (var image = new Image<Rgba32>(2, 2))
            {
                image[0, 0] = new Rgba32(255, 0, 0);
                image[1, 0] = new Rgba32(0, 255, 0);
                image[0, 1] = new Rgba32(0, 0, 255);
                image[1, 1] = new Rgba32(255, 255, 255);
                image.SaveAsPng(imagePath);
            }

            var preprocessor = new ImagePreprocessor();

            PreprocessingResult result = preprocessor.Preprocess(imagePath, 2, 2);

            result.Tensor.Dimensions.ToArray().Should().Equal(1, 3, 2, 2);
            result.Letterbox.Scale.Should().Be(1f);
            result.Letterbox.PadX.Should().Be(0f);
            result.Letterbox.PadY.Should().Be(0f);

            result.Tensor[0, 0, 0, 0].Should().BeApproximately(1f, 0.001f);
            result.Tensor[0, 1, 0, 0].Should().BeApproximately(0f, 0.001f);
            result.Tensor[0, 2, 0, 0].Should().BeApproximately(0f, 0.001f);
            result.Tensor[0, 0, 0, 1].Should().BeApproximately(0f, 0.001f);
            result.Tensor[0, 1, 0, 1].Should().BeApproximately(1f, 0.001f);
            result.Tensor[0, 2, 0, 1].Should().BeApproximately(0f, 0.001f);
            result.Tensor[0, 0, 1, 0].Should().BeApproximately(0f, 0.001f);
            result.Tensor[0, 1, 1, 0].Should().BeApproximately(0f, 0.001f);
            result.Tensor[0, 2, 1, 0].Should().BeApproximately(1f, 0.001f);
        }
        finally
        {
            File.Delete(imagePath);
        }
    }

    [Fact]
    public void PreprocessPreservesLetterboxMetadataForCoordinateRestoration()
    {
        string imagePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");

        try
        {
            using (var image = new Image<Rgba32>(4, 2))
            {
                image.SaveAsPng(imagePath);
            }

            var preprocessor = new ImagePreprocessor();

            PreprocessingResult result = preprocessor.Preprocess(imagePath, 8, 8);

            result.Letterbox.OriginalWidth.Should().Be(4);
            result.Letterbox.OriginalHeight.Should().Be(2);
            result.Letterbox.InputWidth.Should().Be(8);
            result.Letterbox.InputHeight.Should().Be(8);
            result.Letterbox.Scale.Should().Be(2f);
            result.Letterbox.PadX.Should().Be(0f);
            result.Letterbox.PadY.Should().Be(2f);

            const float expectedPadding = 114f / 255f;
            result.Tensor[0, 0, 0, 0].Should().BeApproximately(expectedPadding, 0.001f);
            result.Tensor[0, 1, 0, 0].Should().BeApproximately(expectedPadding, 0.001f);
            result.Tensor[0, 2, 0, 0].Should().BeApproximately(expectedPadding, 0.001f);
        }
        finally
        {
            File.Delete(imagePath);
        }
    }
}
