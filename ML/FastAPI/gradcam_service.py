import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

from config import IMAGE_SIZE


class GradCamService:
    def __init__(self, model):
        self.model = model
        self.last_conv_layer_name = self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name

        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.Model):
                for nested_layer in reversed(layer.layers):
                    if isinstance(nested_layer, keras.layers.Conv2D):
                        return nested_layer.name

        raise ValueError("No Conv2D layer found in model.")

    def make_heatmap(self, input_batch, predicted_class_index: int):
        last_conv_layer = self.model.get_layer(self.last_conv_layer_name)

        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[last_conv_layer.output, self.model.output]
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

    def create_overlay(self, original_image, heatmap, alpha=0.4):
        original_height, original_width = original_image.shape[:2]

        heatmap_resized = cv2.resize(
            heatmap,
            (original_width, original_height)
        )

        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            original_image.astype("uint8"),
            1 - alpha,
            colored_heatmap,
            alpha,
            0
        )

        return overlay