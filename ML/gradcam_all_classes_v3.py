#!/usr/bin/env python3
"""
Grad-CAM visualization for all classes in a grid (v3).

This script generates a 3x3 grid showing Grad-CAM heatmaps for one image across all 9 classes,
showing what the model focuses on when trying to classify the image as each specific class.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def visualize_gradcam_all_classes(
    img_path,
    model,
    class_names,
    target_layer_name=None,
    alpha=0.4,
    save_path=None
):
    """
    Create a 3x3 grid showing Grad-CAM heatmaps for one image across all 9 classes.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list
        List of class names (must have 9 elements)
    target_layer_name : str, optional
        Name of the layer to use for Grad-CAM. If None, auto-detects last conv layer.
    alpha : float
        Transparency of heatmap overlay (0.0-1.0)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    dict: {"predicted_class": int, "probabilities": np.array, "heatmaps": list}
    """
    # Validate inputs
    if len(class_names) != 9:
        raise ValueError(f"Expected 9 class names, got {len(class_names)}")

    # Find last conv layer if not specified
    if target_layer_name is None:
        target_layer_name = find_last_conv_layer(model)

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    img_tensor = np.expand_dims(arr, axis=0)

    # Get predictions
    probs = model.predict(img_tensor, verbose=0)[0]
    predicted_class = int(np.argmax(probs))

    print(f"Model predicts: {class_names[predicted_class]} (class {predicted_class}) with {probs[predicted_class]:.2%} confidence")

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    heatmaps = []

    # Generate Grad-CAM for all 9 classes
    for class_idx in range(9):
        # Generate heatmap for this class
        heatmap = make_gradcam_heatmap(img_tensor, model, target_layer_name, pred_index=class_idx)
        heatmaps.append(heatmap)

        # Resize heatmap to image size
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_resized = cv2.resize(heatmap_resized, (224, 224))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = (
            np.array(img).astype(np.float32) * (1 - alpha) +
            heatmap_colored.astype(np.float32) * alpha
        ).astype(np.uint8)

        # Plot
        axes[class_idx].imshow(overlay)
        axes[class_idx].axis('off')

        # Highlight the predicted class
        if class_idx == predicted_class:
            title = f"★ {class_names[class_idx]}\n({probs[class_idx]:.1%})"
            axes[class_idx].set_title(title, fontsize=12, fontweight='bold', color='green')
        else:
            title = f"{class_names[class_idx]}\n({probs[class_idx]:.1%})"
            axes[class_idx].set_title(title, fontsize=11)

    # Add super title
    img_source = img_path.split('/')[-3] if '/' in img_path else 'unknown'
    plt.suptitle(f"Grad-CAM Heatmaps for All Classes\nImage from: {img_source}",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    print(f"\n★ = Predicted class: {class_names[predicted_class]}")

    return {
        "predicted_class": predicted_class,
        "probabilities": probs,
        "heatmaps": heatmaps
    }


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model for Grad-CAM."""
    # For MobileNetV3, the last conv layer is typically 'conv_1'
    # Try to find it by name first
    for layer in reversed(model.layers):
        if 'conv_1' in layer.name:
            return layer.name

    # Fallback: find last Conv2D layer
    for layer in reversed(model.layers):
        if 'Conv2D' in layer.__class__.__name__:
            return layer.name

    raise ValueError("Could not find a suitable convolutional layer for Grad-CAM")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image.

    Args:
        img_array: Preprocessed image array (1, 224, 224, 3)
        model: The full model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Class index to visualize (if None, uses the top prediction)

    Returns:
        heatmap: Grad-CAM heatmap as numpy array
    """
    # Create a model that maps input -> last conv layer output + predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of predicted class w.r.t. last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the class score w.r.t. feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
