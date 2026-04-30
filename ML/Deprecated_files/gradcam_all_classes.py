def visualize_gradcam_all_classes(
    img_path,
    model,
    class_names,
    target_layer_name="mobilenetv2_1.00_224",
    alpha=0.45,
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
    target_layer_name : str
        Name of the layer to use for Grad-CAM (default: "mobilenetv2_1.00_224")
    alpha : float
        Transparency of heatmap overlay (0.0-1.0)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    dict: {"predicted_class": int, "probabilities": np.array, "heatmaps": list}
    """
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    import matplotlib

    # Validate inputs
    if len(class_names) != 9:
        raise ValueError(f"Expected 9 class names, got {len(class_names)}")

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    img_tensor = np.expand_dims(arr, axis=0)

    # Get predictions
    probs = model.predict(img_tensor, verbose=0)[0]
    predicted_class = int(np.argmax(probs))

    print(f"Model predicts: {class_names[predicted_class]} (class {predicted_class}) with {probs[predicted_class]:.2%} confidence")

    # Get model layers
    mobilenet_layer = model.get_layer(target_layer_name)
    subsequent_layers = model.layers[model.layers.index(mobilenet_layer) + 1:]

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    heatmaps = []
    cmap = matplotlib.colormaps['jet']

    # Generate Grad-CAM for all 9 classes
    for class_idx in range(9):
        # Compute Grad-CAM for this specific class
        with tf.GradientTape() as tape:
            conv_outputs = mobilenet_layer(img_tensor, training=False)
            tape.watch(conv_outputs)

            x = conv_outputs
            for layer in subsequent_layers:
                x = layer(x, training=False)
            predictions = x

            class_score = predictions[:, class_idx]

        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        conv_outputs_np = conv_outputs[0]
        pooled_grads_np = pooled_grads[0]

        heatmap = tf.reduce_sum(conv_outputs_np * pooled_grads_np, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmaps.append(heatmap)

        # Resize and overlay
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size)
        )

        colored = cmap(heatmap_resized / 255.0)
        colored = (colored[..., :3] * 255).astype(np.uint8)

        overlay = (
            np.array(img).astype(np.float32) * (1 - alpha) +
            colored.astype(np.float32) * alpha
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
        import os
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
