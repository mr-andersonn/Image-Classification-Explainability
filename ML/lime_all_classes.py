def visualize_lime_all_classes(
    img_path,
    model,
    class_names,
    num_samples=500,
    num_features=10,
    save_path=None
):
    """
    Create a 3x3 grid showing LIME explanations for one image across all 9 classes.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list
        List of class names (must have 9 elements)
    num_samples : int
        Number of perturbed samples for LIME (default: 500, lower for speed)
    num_features : int
        Number of superpixels to highlight (default: 10)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    dict: {"predicted_class": int, "probabilities": np.array, "explanations": list}
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from skimage.segmentation import quickshift
    from sklearn.linear_model import Ridge
    from sklearn.metrics import pairwise_distances

    # Validate inputs
    if len(class_names) != 9:
        raise ValueError(f"Expected 9 class names, got {len(class_names)}")

    # Load image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32")

    # Get predictions
    preprocessed = preprocess_input(img_array.copy())
    img_tensor = np.expand_dims(preprocessed, axis=0)
    probs = model.predict(img_tensor, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    print(f"Predicted: {class_names[pred_class]} (class {pred_class}) prob={probs[pred_class]*100:.2f}%")

    # Create superpixels once (reuse for all classes)
    segments = quickshift(img_array / 255.0, kernel_size=4, max_dist=200, ratio=0.2)
    num_segments = len(np.unique(segments))
    print(f"Created {num_segments} superpixels")

    # Helper functions
    def perturb_image(img, segments, active_segments):
        perturbed = img.copy()
        for seg_id in range(len(np.unique(segments))):
            if seg_id not in active_segments:
                perturbed[segments == seg_id] = 0
        return perturbed

    # Generate perturbations once
    perturbations = np.random.randint(0, 2, size=(num_samples, num_segments))

    # Get predictions for all perturbations (batch process)
    print(f"Generating {num_samples} perturbations for all classes...")
    all_predictions = []

    for i in range(num_samples):
        active_segments = np.where(perturbations[i] == 1)[0]
        perturbed_img = perturb_image(img_array, segments, active_segments)
        preprocessed_perturb = preprocess_input(perturbed_img.copy())
        perturbed_tensor = np.expand_dims(preprocessed_perturb, axis=0)
        pred = model.predict(perturbed_tensor, verbose=0)[0]  # All classes
        all_predictions.append(pred)

    all_predictions = np.array(all_predictions)  # (num_samples, 9)

    # Calculate distances and weights once
    original_representation = np.ones(num_segments)
    distances = pairwise_distances(
        perturbations,
        original_representation.reshape(1, -1),
        metric='cosine'
    ).ravel()

    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    explanations = []

    # Generate LIME for all 9 classes
    for class_idx in range(9):
        print(f"  Processing class {class_idx}: {class_names[class_idx]}...")

        # Get predictions for this class
        predictions = all_predictions[:, class_idx]

        # Fit interpretable model
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

        explanations.append(explanation_mask)

        # Plot
        axes[class_idx].imshow(explanation_mask, cmap='jet')
        axes[class_idx].axis('off')

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
        f"LIME Explanations (All 9 Classes)\nImage from: {img_source}",
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
        "explanations": explanations
    }
