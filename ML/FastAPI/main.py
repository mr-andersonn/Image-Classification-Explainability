from fastapi import FastAPI

from model_service import ModelService
from gradcam_service import GradCamService
from activation_service import ActivationService
from routes import create_router


app = FastAPI(title="Fish Classifier Explainability API")

model_service = ModelService()
gradcam_service = GradCamService(model_service.model)
activation_service = ActivationService(model_service.model)

router = create_router(
    model_service,
    gradcam_service,
    activation_service
)

app.include_router(router)