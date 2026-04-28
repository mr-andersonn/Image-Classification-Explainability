from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from config import IMAGE_SIZE
from image_utils import load_image_from_bytes, image_to_base64


def create_router(model_service, gradcam_service, activation_service):
    router = APIRouter()

    @router.get("/")
    def root():
        return {
            "message": "Fish classifier explainability API is running."
        }

    @router.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image."
            )

        image_bytes = await file.read()

        original_image, input_batch = load_image_from_bytes(
            image_bytes,
            IMAGE_SIZE
        )

        prediction_result = model_service.predict(input_batch)

        heatmap = gradcam_service.make_heatmap(
            input_batch,
            prediction_result["predicted_index"]
        )

        overlay = gradcam_service.create_overlay(original_image, heatmap)
        overlay_base64 = image_to_base64(overlay)

        activation_grids = activation_service.generate_activation_grids(
            input_batch,
            images_per_row=16,
            max_layers=5,
            max_channels=64
        )

        activation_images = [
            {
                "layerName": item["layerName"],
                "image": image_to_base64(item["image"])
            }
            for item in activation_grids
        ]

        return JSONResponse(
            content={
                "predictedClass": prediction_result["predicted_class"],
                "predictedIndex": prediction_result["predicted_index"],
                "confidence": prediction_result["confidence"],
                "allPredictions": prediction_result["all_predictions"],
                "heatmapImage": overlay_base64,
                "activationImages": activation_images
            }
        )

    return router