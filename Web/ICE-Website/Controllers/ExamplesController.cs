using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using ICE_Website.Models.Examples;
using ICE_Website.Models;

namespace ICE_Website.Controllers;

public class ExamplesController : Controller
{
    private readonly HttpClient _httpClient;
    private readonly IWebHostEnvironment _environment;

    public ExamplesController(
        IHttpClientFactory httpClientFactory,
        IWebHostEnvironment environment)
    {
        _httpClient = httpClientFactory.CreateClient();
        _environment = environment;
    }

    [HttpGet]
    public IActionResult Index()
    {
        return View(CreatePageModel());
    }

    [HttpPost]
    public async Task<IActionResult> Index(string selectedImageUrl)
    {
        var model = CreatePageModel();
        model.SelectedImageUrl = selectedImageUrl;

        try
        {
            byte[] imageBytes;
            string fileName;

            if (selectedImageUrl.StartsWith("http"))
            {
                imageBytes = await _httpClient.GetByteArrayAsync(selectedImageUrl);
                fileName = "internet-image.jpg";
            }
            else
            {
                var relativePath = selectedImageUrl.TrimStart('/');
                var fullPath = Path.Combine(_environment.WebRootPath, relativePath);

                imageBytes = await System.IO.File.ReadAllBytesAsync(fullPath);
                fileName = Path.GetFileName(fullPath);
            }

            using var form = new MultipartFormDataContent();
            using var imageContent = new ByteArrayContent(imageBytes);

            imageContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");

            form.Add(imageContent, "file", fileName);

            var response = await _httpClient.PostAsync(
                "http://127.0.0.1:8000/predict",
                form
            );

            if (!response.IsSuccessStatusCode)
            {
                model.ErrorMessage = "The ML service returned an error.";
                return View(model);
            }

            var json = await response.Content.ReadAsStringAsync();

            var prediction = JsonSerializer.Deserialize<PredictionResult>(
                json,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                }
            );

            if (prediction == null)
            {
                model.ErrorMessage = "Could not read prediction result.";
                return View(model);
            }

            model.PredictionResult = prediction;

            return View(model);
        }
        catch (Exception ex)
        {
            model.ErrorMessage = $"Error: {ex.Message}";
            return View(model);
        }
    }

    private ExamplesPageModel CreatePageModel()
    {
        return new ExamplesPageModel
        {
            Images = new List<ExampleImage>
            {
                new()
                {
                    Title = "Black Sea Sprat",
                    ImageUrl = "/images/dataset/ds_black_sea_sprat.png",
                    SourceType = "Dataset"
                },
                new()
                {
                    Title = "Mackerel",
                    ImageUrl = "/images/dataset/ds_hourse_mackerel.png",
                    SourceType = "Dataset"
                },
                new()
                {
                    Title = "Shrimp",
                    ImageUrl = "/images/dataset/ds_shrimp.png",
                    SourceType = "Dataset"
                },
                new()
                {
                    Title = "Angelfish",
                    ImageUrl = "/images/random/random_angelfish.png",
                    SourceType = "Random"
                },
                new()
                {
                    Title = "Nemo",
                    ImageUrl = "/images/random/random_nemo.jpg",
                    SourceType = "Random"
                },
                new()
                {
                    Title = "BMW",
                    ImageUrl = "/images/random/random_bmw.png",
                    SourceType = "Random"
                },
                new()
                {
                    Title = "Aquaman",
                    ImageUrl = "/images/random/random_aquaman.jpg",
                    SourceType = "Random"
                },
            }
        };
    }
}