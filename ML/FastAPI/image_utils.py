import io
import base64
import numpy as np
from PIL import Image


def load_image_from_bytes(image_bytes: bytes, image_size: tuple[int, int]):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    original_image = np.array(image)

    resized_image = image.resize(image_size)
    input_array = np.array(resized_image).astype("float32")
    input_batch = np.expand_dims(input_array, axis=0)

    return original_image, input_batch


def image_to_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array.astype("uint8"))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"