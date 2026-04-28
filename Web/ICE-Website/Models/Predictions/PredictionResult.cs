namespace ICE_Website.Models;

public class PredictionResult
{
    public string PredictedClass { get; set; } = "";

    public int PredictedIndex { get; set; }

    public double Confidence { get; set; }

    public List<ClassPrediction> AllPredictions { get; set; } = new();

    public string HeatmapImage { get; set; } = "";

    public List<ActivationImage> ActivationImages { get; set; } = new();
}