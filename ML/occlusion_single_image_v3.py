def visualize_occlusion(
    img_path,
    model,
    class_names=None,
    target_class_idx=None,
    patch_size=28,
    stride=14,
    alpha=0.45,
    save_path=None
):
    """
    Create Occlusion Sensitivity visualization for a single image.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list, optional
        List of class names for display. If None, uses class indices.
    target_class_idx : int, optional
        Target class index for occlusion. If None, uses predicted class.
    patch_size : int
        Size of the occlusion patch (default: 28)
    stride : int
        Stride for sliding the occlusion patch (default: 14)
    alpha : float
        Transparency of heatmap overlay (0.0-1.0)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    tuple: (pil_img, sensitivity_map, overlay_img, target_class_idx)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32")

    # Get baseline prediction (no preprocessing - model has built-in rescaling)
    img_tensor = np.expand_dims(img_array, axis=0)
    baseline_preds = model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(baseline_preds))

    if target_class_idx is None:
        target_class_idx = pred_idx

    baseline_score = baseline_preds[target_class_idx]

    print(f"Analyzing occlusion sensitivity for class: {class_names[target_class_idx] if class_names else target_class_idx}")
    print(f"Predicted: {class_names[pred_idx] if class_names else pred_idx} ({baseline_preds[pred_idx]*100:.2f}%)")
    print(f"Baseline score for target class: {baseline_score:.4f}")

    # Create sensitivity map
    img_height, img_width = 224, 224
    sensitivity_map = np.zeros((img_height, img_width))

    # Calculate number of patches
    num_patches_h = (img_height - patch_size) // stride + 1
    num_patches_w = (img_width - patch_size) // stride + 1
    total_patches = num_patches_h * num_patches_w

    print(f"Occluding with {patch_size}x{patch_size} patches, stride={stride}")
    print(f"Testing {total_patches} positions...")

    patch_count = 0

    # Slide occlusion patch across image
    for i in range(0, img_height - patch_size + 1, stride):
        for j in range(0, img_width - patch_size + 1, stride):
            patch_count += 1

            # Create occluded image (gray patch)
            occluded = img_array.copy()
            occluded[i:i+patch_size, j:j+patch_size, :] = 128  # Gray occlusion

            # Get prediction (no preprocessing - model has built-in rescaling)
            occluded_tensor = np.expand_dims(occluded, axis=0)
            occluded_preds = model.predict(occluded_tensor, verbose=0)[0]
            occluded_score = occluded_preds[target_class_idx]

            # Calculate sensitivity (drop in confidence)
            sensitivity = baseline_score - occluded_score

            # Fill the patch region with this sensitivity value
            sensitivity_map[i:i+patch_size, j:j+patch_size] = np.maximum(
                sensitivity_map[i:i+patch_size, j:j+patch_size],
                sensitivity
            )

            # Progress update
            if patch_count % 50 == 0:
                print(f"  ... processed {patch_count}/{total_patches} patches")

    print(f"Completed all {total_patches} patches")

    # Normalize sensitivity map
    sensitivity_map = np.maximum(sensitivity_map, 0)  # Only positive contributions
    if sensitivity_map.max() > 0:
        sensitivity_map = sensitivity_map / sensitivity_map.max()

    # Create overlay
    heatmap_uint8 = (sensitivity_map * 255).astype(np.uint8)
    plt_heat = plt.cm.jet(heatmap_uint8 / 255.0)[:, :, :3]
    plt_heat = (plt_heat * 255).astype(np.uint8)

    overlay = (
        np.array(img).astype(np.float32) * (1 - alpha) +
        plt_heat.astype(np.float32) * alpha
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(sensitivity_map, cmap='jet')
    axes[1].set_title(f'Occlusion Sensitivity\n(patch={patch_size}x{patch_size}, stride={stride})', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay_img)
    if class_names:
        overlay_title = f'Overlay\nPrediction: {class_names[pred_idx]}\nConfidence: {baseline_preds[pred_idx]*100:.2f}%'
    else:
        overlay_title = f'Overlay\nClass {pred_idx}\nConfidence: {baseline_preds[pred_idx]*100:.2f}%'
    axes[2].set_title(overlay_title, fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return img, sensitivity_map, overlay_img, target_class_idx
