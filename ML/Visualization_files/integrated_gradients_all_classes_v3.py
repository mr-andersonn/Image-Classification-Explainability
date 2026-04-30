def visualize_integrated_gradients_all_classes(
    img_path,
    model,
    class_names,
    steps=32,
    alpha=0.45,
    save_path=None
):
    """
    Create a 3x3 grid showing Integrated Gradients heatmaps for one image across all 9 classes.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list
        List of class names (must have 9 elements)
    steps : int
        Number of steps for integrated gradients (default: 32)
    alpha : float
        Transparency of heatmap overlay (0.0-1.0) - not used in heatmap-only view
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

    # Validate inputs
    if len(class_names) != 9:
        raise ValueError(f"Expected 9 class names, got {len(class_names)}")

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    img_tensor = np.expand_dims(arr, axis=0)

    # Get predictions
    probs = model.predict(img_tensor, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    print(f"Predicted: {class_names[pred_class]} (class {pred_class}) prob={probs[pred_class]*100:.2f}%")

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    heatmaps = []
    baseline = tf.zeros_like(img_tensor)

    # Generate Integrated Gradients for all 9 classes
    for class_idx in range(9):
        # Compute Integrated Gradients for this specific class
        alphas = tf.linspace(0.0, 1.0, steps + 1)

        grads_list = []
        for alpha_val in alphas:
            x = baseline + alpha_val * (img_tensor - baseline)
            with tf.GradientTape() as tape:
                tape.watch(x)
                predictions = model(x, training=False)
                score = predictions[:, class_idx]
            grads = tape.gradient(score, x)
            grads_list.append(grads)

        grads = tf.stack(grads_list, axis=0)  # (steps+1, 1, H, W, 3)
        grads_mid = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads_mid, axis=0)  # (1, H, W, 3)

        ig_attributions = (img_tensor - baseline) * avg_grads
        ig_attributions = ig_attributions[0].numpy()  # (H, W, 3)

        # Convert to heatmap
        heatmap = np.sum(np.abs(ig_attributions), axis=-1)  # (H, W)
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmaps.append(heatmap)

        # Plot heatmap
        axes[class_idx].imshow(heatmap, cmap="jet")
        axes[class_idx].axis("off")

        # Create title
        title = f"{class_names[class_idx]}\n{probs[class_idx]*100:.2f}%"
        if class_idx == pred_class:
            title = "★ " + title
            axes[class_idx].set_title(
                title,
                fontsize=12,
                fontweight="bold",
                color="green"
            )
        else:
            axes[class_idx].set_title(
                title,
                fontsize=11,
                fontweight="normal"
            )

    # Add super title
    img_source = img_path.split('/')[-3] if '/' in img_path else 'unknown'
    plt.suptitle(
        f"Integrated Gradients Heatmaps (All 9 Classes)\nImage from: {img_source}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()

    # Save if requested
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    print(f"\n★ = Predicted class: {class_names[pred_class]}")

    return {
        "predicted_class": pred_class,
        "probabilities": probs,
        "heatmaps": heatmaps
    }
