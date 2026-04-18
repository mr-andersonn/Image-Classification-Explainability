def visualize_integrated_gradients(
    img_path,
    model,
    class_names=None,
    target_class_idx=None,
    steps=32,
    alpha=0.45,
    save_path=None
):
    """
    Create Integrated Gradients visualization for a single image.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list, optional
        List of class names for display. If None, uses class indices.
    target_class_idx : int, optional
        Target class index for IG. If None, uses predicted class.
    steps : int
        Number of steps for integrated gradients (default: 32)
    alpha : float
        Transparency of heatmap overlay (0.0-1.0)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    tuple: (pil_img, heatmap, overlay_img, target_class_idx)
    """
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    img_tensor = np.expand_dims(arr, axis=0)

    # Get predictions
    preds = model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))

    if target_class_idx is None:
        target_class_idx = pred_idx

    # Compute Integrated Gradients
    baseline = tf.zeros_like(img_tensor)
    alphas = tf.linspace(0.0, 1.0, steps + 1)

    grads_list = []
    for alpha_val in alphas:
        x = baseline + alpha_val * (img_tensor - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = model(x, training=False)
            score = predictions[:, target_class_idx]
        grads = tape.gradient(score, x)
        grads_list.append(grads)

    grads = tf.stack(grads_list, axis=0)  # (steps+1, 1, H, W, 3)
    grads_mid = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads_mid, axis=0)  # (1, H, W, 3)

    ig_attributions = (img_tensor - baseline) * avg_grads  # (1, H, W, 3)
    ig_attributions = ig_attributions[0].numpy()  # (H, W, 3)

    # Convert to heatmap
    heatmap = np.sum(np.abs(ig_attributions), axis=-1)  # (H, W)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Create overlay
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    plt_heat = plt.cm.jet(heatmap_uint8 / 255.0)[:, :, :3]  # (H, W, 3) RGB
    plt_heat = (plt_heat * 255).astype(np.uint8)

    overlay = (
        np.array(img).astype(np.float32) * (1 - alpha) +
        plt_heat.astype(np.float32) * alpha
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].axis("off")
    if class_names:
        axes[0].set_title(f"Original\nPred: {class_names[pred_idx]} ({preds[pred_idx]*100:.1f}%)", fontsize=14)
    else:
        axes[0].set_title(f"Original\nPred: Class {pred_idx} ({preds[pred_idx]*100:.1f}%)", fontsize=14)

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].axis("off")
    if class_names:
        axes[1].set_title(f"Integrated Gradients Heatmap\nClass: {class_names[target_class_idx]}", fontsize=14)
    else:
        axes[1].set_title(f"Integrated Gradients Heatmap\nClass: {target_class_idx}", fontsize=14)

    axes[2].imshow(overlay_img)
    axes[2].axis("off")
    axes[2].set_title("Overlay", fontsize=14)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return img, heatmap, overlay_img, target_class_idx
