from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
import base64
import os

app = FastAPI(title="Fish Classifier Explainability API")

MODEL_PATH = "MobileNetV3Large_Improved.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout",
]

model = keras.models.load_model(MODEL_PATH)


def load_image_from_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_image = np.array(image)

    resized_image = image.resize(IMAGE_SIZE)
    input_array = np.array(resized_image).astype("float32")

    # Shape: (1, 224, 224, 3)
    input_batch = np.expand_dims(input_array, axis=0)

    return original_image, input_batch


def predict_image(input_batch):
    predictions = model.predict(input_batch, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    all_predictions = [
        {
            "className": CLASS_NAMES[i],
            "confidence": float(predictions[i])
        }
        for i in range(len(CLASS_NAMES))
    ]

    return predicted_index, predicted_class, confidence, all_predictions


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name

    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, keras.layers.Conv2D):
                    return nested_layer.name

    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(input_batch, predicted_class_index):
    last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_batch)
        class_score = predictions[:, predicted_class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        return np.zeros(IMAGE_SIZE, dtype=np.float32)

    heatmap /= max_value

    return heatmap.numpy()


def create_overlay(original_image, heatmap, alpha=0.4):
    original_height, original_width = original_image.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (original_width, original_height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image.astype("uint8"),
        1 - alpha,
        colored_heatmap,
        alpha,
        0
    )

    return overlay


def image_to_base64(image_array):
    image = Image.fromarray(image_array.astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/")
def root():
    return {
        "message": "Fish classifier explainability API is running."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    original_image, input_batch = load_image_from_bytes(image_bytes)

    predicted_index, predicted_class, confidence, all_predictions = predict_image(input_batch)

    heatmap = make_gradcam_heatmap(input_batch, predicted_index)
    overlay = create_overlay(original_image, heatmap)
    overlay_base64 = image_to_base64(overlay)

    return JSONResponse(
        content={
            "predictedClass": predicted_class,
            "predictedIndex": predicted_index,
            "confidence": confidence,
            "allPredictions": all_predictions,
            "heatmapImage": overlay_base64
        }
    )