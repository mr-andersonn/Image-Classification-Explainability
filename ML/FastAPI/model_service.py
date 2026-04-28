import numpy as np
from tensorflow import keras

from config import MODEL_PATH, CLASS_NAMES


class ModelService:
    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)

    def predict(self, input_batch):
        predictions = self.model.predict(input_batch, verbose=0)[0]

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

        return {
            "predicted_index": predicted_index,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions
        }