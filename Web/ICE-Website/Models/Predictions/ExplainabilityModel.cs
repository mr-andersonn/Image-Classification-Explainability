namespace ICE_Website.Models;

public class ExplainabilityModel
{
    public IFormFile? ImageFile { get; set; }

    public string? PredictedClass { get; set; }

    public int? PredictedIndex { get; set; }

    public double? Confidence { get; set; }

    public string? HeatmapImage { get; set; }

    public List<ClassPrediction> AllPredictions { get; set; } = new();

    public List<ActivationImage> ActivationImages { get; set; } = new();

    public string? ErrorMessage { get; set; }
}