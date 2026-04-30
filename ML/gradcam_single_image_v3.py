#!/usr/bin/env python3
"""
Grad-CAM visualization for single image classification.

This script generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations
for a single image using a trained model. It shows which parts of the image the model
focuses on when making predictions.

Usage:
    python gradcam_single_image_v3.py <image_path> <model_path> <save_path> [--class CLASS_NAME]

Example:
    python gradcam_single_image_v3.py image.png Models/model.keras output.png
    python gradcam_single_image_v3.py image.png Models/model.keras output.png --class "Red Mullet"
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Class names for the fish dataset
CLASS_NAMES = [
    "Black Sea Sprat",      # 0
    "Gilt-Head Bream",      # 1
    "Hourse Mackerel",      # 2
    "Red Mullet",           # 3
    "Red Sea Bream",        # 4
    "Sea Bass",             # 5
    "Shrimp",               # 6
    "Striped Red Mullet",   # 7
    "Trout",                # 8
]

def load_and_preprocess_image(path, target_size=(224, 224)):
    """Load and preprocess image for model input."""
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")
    return arr

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

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay the heatmap on the original image.

    Args:
        img_path: Path to original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use

    Returns:
        original_img: Original image
        heatmap_colored: Colored heatmap
        superimposed_img: Image with heatmap overlay
    """
    # Load original image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)

    # Rescale heatmap to 0-255 and resize to image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on image
    superimposed_img = heatmap_colored * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return img, heatmap_colored, superimposed_img

def visualize_gradcam(
    img_path,
    model,
    class_names=None,
    target_layer_name=None,
    class_index=None,
    alpha=0.4,
    save_path=None,
    true_class_index=None
):
    """
    Create Grad-CAM visualization for a single image (compatible with old API).

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list, optional
        List of class names for display. If None, uses class indices.
    target_layer_name : str, optional
        Name of the layer to use for Grad-CAM. If None, auto-detects last conv layer.
    class_index : int, optional
        Class index to generate CAM for. If None, uses predicted class.
    alpha : float
        Transparency of heatmap overlay (0.0-1.0)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    tuple: (pil_img, heatmap, overlay_img, used_class_index)
    """
    # Find last conv layer if not specified
    if target_layer_name is None:
        target_layer_name = find_last_conv_layer(model)

    # Load and preprocess image
    img_array = load_and_preprocess_image(img_path)
    img_batch = np.expand_dims(img_array, axis=0)

    # Get predictions
    predictions = model.predict(img_batch, verbose=0)[0]

    # Determine class index
    if class_index is None:
        class_index = int(np.argmax(predictions))

    class_prob = float(predictions[class_index])

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_batch, model, target_layer_name, pred_index=class_index)

    # Overlay on image
    original, heatmap_colored, superimposed = overlay_heatmap_on_image(img_path, heatmap, alpha=alpha)

    # Convert to PIL Images for return
    original_pil = Image.fromarray(original)
    overlay_pil = Image.fromarray(superimposed)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay_pil)
    # Create title with prediction, confidence, and true class
    if class_names:
        overlay_title = f'Overlay\nPrediction: {class_names[class_index]}\nConfidence: {class_prob*100:.2f}%'
        if true_class_index is not None:
            overlay_title += f'\nTrue: {class_names[true_class_index]}'
    else:
        overlay_title = f'Overlay\nClass {class_index}\nConfidence: {class_prob*100:.2f}%'
        if true_class_index is not None:
            overlay_title += f'\nTrue Class: {true_class_index}'
    axes[2].set_title(overlay_title, fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return original_pil, heatmap, overlay_pil, class_index

def main():
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualization for a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize prediction for predicted class (default)
  python gradcam_single_image_v3.py fish.png Models/MobileNetV3Large_Improved.keras output.png

  # Visualize for specific class
  python gradcam_single_image_v3.py fish.png Models/MobileNetV3Large_Improved.keras output.png --class "Red Mullet"
        """
    )

    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('model_path', type=str, help='Path to trained model (.keras file)')
    parser.add_argument('save_path', type=str, help='Path to save output visualization')
    parser.add_argument('--class', dest='class_name', type=str, default=None,
                        help='Class name to visualize (if not provided, uses predicted class)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Transparency of heatmap overlay (default: 0.4)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    # Find last conv layer
    last_conv_layer = find_last_conv_layer(model)
    print(f"Using convolutional layer: {last_conv_layer}")

    # Load and preprocess image
    print(f"Loading image from {args.image_path}...")
    img_array = load_and_preprocess_image(args.image_path)
    img_array_batch = np.expand_dims(img_array, axis=0)

    # Get prediction
    predictions = model.predict(img_array_batch, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    pred_class = CLASS_NAMES[pred_idx]
    pred_prob = float(predictions[pred_idx])

    print(f"Model predicts: {pred_class} (class {pred_idx}) with {pred_prob*100:.2f}% confidence")

    # Determine target class for visualization
    if args.class_name is not None:
        if args.class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class: {args.class_name}. Valid classes: {CLASS_NAMES}")
        target_idx = CLASS_NAMES.index(args.class_name)
        target_class = args.class_name
        target_prob = float(predictions[target_idx])
        print(f"Visualizing for class: {target_class} (class {target_idx}) with {target_prob*100:.2f}% confidence")
    else:
        target_idx = pred_idx
        target_class = pred_class
        target_prob = pred_prob
        print(f"Visualizing for predicted class: {target_class}")

    # Generate Grad-CAM heatmap
    print("Generating Grad-CAM heatmap...")
    heatmap = make_gradcam_heatmap(img_array_batch, model, last_conv_layer, pred_index=target_idx)

    # Overlay on image
    original, heatmap_colored, superimposed = overlay_heatmap_on_image(
        args.image_path, heatmap, alpha=args.alpha
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(heatmap_colored)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title(f'Overlay\n{target_class} ({target_prob*100:.2f}%)', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.save_path}")

    # Also display if running interactively
    plt.show()

    print("✓ Grad-CAM visualization complete!")

if __name__ == '__main__':
    main()
