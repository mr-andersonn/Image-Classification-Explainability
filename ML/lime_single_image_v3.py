def visualize_lime(
    img_path,
    model,
    class_names=None,
    target_class_idx=None,
    num_samples=1000,
    num_features=10,
    alpha=0.45,
    save_path=None
):
    """
    Create LIME visualization for a single image.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list, optional
        List of class names for display. If None, uses class indices.
    target_class_idx : int, optional
        Target class index for LIME. If None, uses predicted class.
    num_samples : int
        Number of perturbed samples for LIME (default: 1000)
    num_features : int
        Number of superpixels to highlight (default: 10)
    alpha : float
        Transparency of heatmap overlay (0.0-1.0)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    tuple: (pil_img, explanation_mask, overlay_img, target_class_idx)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage.segmentation import quickshift
    from sklearn.linear_model import Ridge
    from sklearn.metrics import pairwise_distances

    # Load image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32")

    # Get predictions (no preprocessing - model has built-in rescaling)
    img_tensor = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))

    if target_class_idx is None:
        target_class_idx = pred_idx

    print(f"Explaining prediction for class: {class_names[target_class_idx] if class_names else target_class_idx}")
    print(f"Predicted: {class_names[pred_idx] if class_names else pred_idx} ({preds[pred_idx]*100:.2f}%)")

    # Create superpixels
    segments = quickshift(img_array / 255.0, kernel_size=4, max_dist=200, ratio=0.2)
    num_segments = len(np.unique(segments))
    print(f"Created {num_segments} superpixels")

    # Generate perturbed samples
    def perturb_image(img, segments, active_segments):
        """Create perturbed image by hiding inactive segments"""
        perturbed = img.copy()
        for seg_id in range(len(np.unique(segments))):
            if seg_id not in active_segments:
                perturbed[segments == seg_id] = 0  # Black out inactive segments
        return perturbed

    # Create random perturbations
    num_perturb = num_samples
    perturbations = np.random.randint(0, 2, size=(num_perturb, num_segments))

    # Get predictions for perturbed images
    predictions = []
    print(f"Generating {num_perturb} perturbations...")

    for i in range(num_perturb):
        active_segments = np.where(perturbations[i] == 1)[0]
        perturbed_img = perturb_image(img_array, segments, active_segments)

        # Predict (no preprocessing - model has built-in rescaling)
        perturbed_tensor = np.expand_dims(perturbed_img, axis=0)
        pred = model.predict(perturbed_tensor, verbose=0)[0][target_class_idx]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate distances (similarity between original and perturbed)
    original_representation = np.ones(num_segments)
    distances = pairwise_distances(
        perturbations,
        original_representation.reshape(1, -1),
        metric='cosine'
    ).ravel()

    # Kernel width for exponential kernel
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    # Fit interpretable model (Ridge regression)
    ridge = Ridge(alpha=1, fit_intercept=True)
    ridge.fit(perturbations, predictions, sample_weight=weights)

    # Get feature importance
    coefs = ridge.coef_

    # Get top features
    top_features = np.argsort(coefs)[-num_features:]

    # Create explanation mask
    explanation_mask = np.zeros(img_array.shape[:2])
    for seg_id in top_features:
        explanation_mask[segments == seg_id] = coefs[seg_id]

    # Normalize mask
    explanation_mask = explanation_mask - explanation_mask.min()
    if explanation_mask.max() > 0:
        explanation_mask = explanation_mask / explanation_mask.max()

    # Create overlay
    heatmap_uint8 = (explanation_mask * 255).astype(np.uint8)
    plt_heat = plt.cm.jet(heatmap_uint8 / 255.0)[:, :, :3]
    plt_heat = (plt_heat * 255).astype(np.uint8)

    overlay = (
        np.array(img).astype(np.float32) * (1 - alpha) +
        plt_heat.astype(np.float32) * alpha
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    true_class_name = img_path.replace("\\", "/").split("/")[-3]

    axes[0].imshow(img)
    axes[0].set_title(f'Original Image\nTrue: {true_class_name}', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(explanation_mask, cmap='jet')
    axes[1].set_title(f'LIME Explanation\n({num_segments} superpixels, top {num_features})', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay_img)
    if class_names:
        overlay_title = f'Overlay\nPrediction: {class_names[pred_idx]}\nConfidence: {preds[pred_idx]*100:.2f}%'
    else:
        overlay_title = f'Overlay\nClass {pred_idx}\nConfidence: {preds[pred_idx]*100:.2f}%'
    axes[2].set_title(overlay_title, fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return img, explanation_mask, overlay_img, target_class_idx
