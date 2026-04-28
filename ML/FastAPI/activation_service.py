import numpy as np
import cv2
from tensorflow import keras


class ActivationService:
    def __init__(self, model):
        self.model = model
        self.layer_names, self.layer_outputs = self._get_visualizable_layers()

        self.activation_model = keras.Model(
            inputs=self.model.input,
            outputs=self.layer_outputs
        )

    def _get_visualizable_layers(self):
        layer_names = []
        layer_outputs = []

        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                layer_names.append(layer.name)
                layer_outputs.append(layer.output)

            elif isinstance(layer, keras.Model):
                for nested_layer in layer.layers:
                    if isinstance(nested_layer, keras.layers.Conv2D):
                        layer_names.append(nested_layer.name)
                        layer_outputs.append(nested_layer.output)

        return layer_names, layer_outputs

    def generate_activation_grids(
        self,
        input_batch,
        images_per_row: int = 16,
        max_layers: int = 5,
        max_channels: int = 64
    ):
        activations = self.activation_model.predict(input_batch, verbose=0)

        if not isinstance(activations, list):
            activations = [activations]

        activation_images = []

        for layer_name, layer_activation in zip(
            self.layer_names[:max_layers],
            activations[:max_layers]
        ):
            grid = self._create_activation_grid(
                layer_activation,
                images_per_row,
                max_channels
            )

            activation_images.append({
                "layerName": layer_name,
                "image": grid
            })

        return activation_images

    def _create_activation_grid(
        self,
        layer_activation,
        images_per_row: int,
        max_channels: int
    ):
        n_features = min(layer_activation.shape[-1], max_channels)
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row

        if n_cols == 0:
            n_cols = 1

        display_grid = np.zeros(
            (
                (size + 1) * n_cols - 1,
                images_per_row * (size + 1) - 1
            )
        )

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_index = col * images_per_row + row

                if channel_index >= n_features:
                    continue

                channel_image = layer_activation[0, :, :, channel_index].copy()

                if channel_image.sum() != 0:
                    channel_image -= channel_image.mean()

                    std = channel_image.std()
                    if std != 0:
                        channel_image /= std

                    channel_image *= 64
                    channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype("uint8")

                display_grid[
                    col * (size + 1): (col + 1) * size + col,
                    row * (size + 1): (row + 1) * size + row,
                ] = channel_image

        display_grid = display_grid.astype("uint8")

        colored_grid = cv2.applyColorMap(display_grid, cv2.COLORMAP_VIRIDIS)
        colored_grid = cv2.cvtColor(colored_grid, cv2.COLOR_BGR2RGB)

        return colored_grid