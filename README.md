# Image-Classification-Explainability
DAT255 Project Group 42.
README med hvordan man kjører trening, visualisering og app
requirements.txt eller environment.yml
tydelig lenke til datasettet
notebooks/scripts navngitt forståelig
lagrede figurer eller kode som genererer figurene
modellfil eller instruksjon for hvordan modellen trenes
ingen store, unødvendige filer eller lokale paths som bare fungerer på én PC

<br>
# Deployment

Kjøring av applikasjonen lokalt innebærer 2 steg:

## 1. Starte FastAPI-serveren
Dette kan gjøres ved å laste ned repository, navigere til følgende mappe:

`Image-Classification-Explainability/ML/FastAPI/`

og starte serveren med kommandoen:

```bash
uvicorn main:app --reload
```

## 2. Starte ASP.NET MVC-applikasjonen
Naviger til følgende mappe: 
`Image-Classification-Explainability/Web/ICE-Website/`

og kjør kommandoen 

```bash
dotnet run
```

Lenken til nettsiden vil bli skrevet ut i terminalen.
