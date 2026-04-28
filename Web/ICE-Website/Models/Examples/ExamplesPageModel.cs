namespace ICE_Website.Models.Examples;

public class ExamplesPageModel
{
    public List<ExampleImage> Images { get; set; } = new();

    public string? SelectedImageUrl { get; set; }

    public PredictionResult? PredictionResult { get; set; }

    public string? ErrorMessage { get; set; }
}