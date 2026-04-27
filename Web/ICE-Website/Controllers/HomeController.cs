using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using ICE_Website.Models;

namespace YourProjectName.Controllers;

public class HomeController : Controller
{
    private readonly HttpClient _httpClient;

    public HomeController(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
    }

    [HttpGet]
    public IActionResult Index()
    {
        return View(new ExplainabilityModel());
    }

    [HttpPost]
    public async Task<IActionResult> Index(ExplainabilityModel model)
    {
        if (model.ImageFile == null || model.ImageFile.Length == 0)
        {
            model.ErrorMessage = "Please upload an image.";
            return View(model);
        }

        try
        {
            using var form = new MultipartFormDataContent();

            using var stream = model.ImageFile.OpenReadStream();
            using var fileContent = new StreamContent(stream);

            fileContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue(model.ImageFile.ContentType);

            form.Add(fileContent, "file", model.ImageFile.FileName);

            var response = await _httpClient.PostAsync("http://127.0.0.1:8000/predict", form);

            if (!response.IsSuccessStatusCode)
            {
                model.ErrorMessage = "The ML service returned an error.";
                return View(model);
            }

            var json = await response.Content.ReadAsStringAsync();

            var result = JsonSerializer.Deserialize<ExplainabilityModel>(
                json,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

            if (result == null)
            {
                model.ErrorMessage = "Could not read prediction result.";
                return View(model);
            }

            model.PredictedClass = result.PredictedClass;
            model.Confidence = result.Confidence;
            model.HeatmapImage = result.HeatmapImage;
            model.AllPredictions = result.AllPredictions;

            return View(model);
        }
        catch (Exception ex)
        {
            model.ErrorMessage = $"Error: {ex.Message}";
            return View(model);
        }
    }
}