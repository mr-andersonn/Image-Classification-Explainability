def visualize_occlusion_all_classes(
    img_path,
    model,
    class_names,
    patch_size=28,
    stride=28,
    save_path=None
):
    """
    Create a 3x3 grid showing Occlusion Sensitivity for one image across all 9 classes.

    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    class_names : list
        List of class names (must have 9 elements)
    patch_size : int
        Size of the occlusion patch (default: 28)
    stride : int
        Stride for sliding the occlusion patch (default: 28, faster)
    save_path : str, optional
        Path to save the visualization. If None, only displays.

    Returns:
    --------
    dict: {"predicted_class": int, "probabilities": np.array, "sensitivity_maps": list}
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Validate inputs
    if len(class_names) != 9:
        raise ValueError(f"Expected 9 class names, got {len(class_names)}")

    # Load image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32")

    # Get baseline prediction
    preprocessed = preprocess_input(img_array.copy())
    img_tensor = np.expand_dims(preprocessed, axis=0)
    baseline_preds = model.predict(img_tensor, verbose=0)[0]
    pred_class = int(np.argmax(baseline_preds))

    print(f"Predicted: {class_names[pred_class]} (class {pred_class}) prob={baseline_preds[pred_class]*100:.2f}%")
    print(f"Occluding with {patch_size}x{patch_size} patches, stride={stride}")

    # Calculate number of patches
    img_height, img_width = 224, 224
    num_patches_h = (img_height - patch_size) // stride + 1
    num_patches_w = (img_width - patch_size) // stride + 1
    total_patches = num_patches_h * num_patches_w

    print(f"Total positions to test: {total_patches}")
    print(f"Total forward passes: {total_patches} (reused for all classes)")

    # Pre-compute all occluded images and their predictions
    all_predictions = []
    patch_positions = []

    print("Pre-computing occlusions...")
    patch_count = 0

    for i in range(0, img_height - patch_size + 1, stride):
        for j in range(0, img_width - patch_size + 1, stride):
            patch_count += 1

            # Create occluded image
            occluded = img_array.copy()
            occluded[i:i+patch_size, j:j+patch_size, :] = 128  # Gray occlusion

            # Get prediction
            preprocessed_occluded = preprocess_input(occluded)
            occluded_tensor = np.expand_dims(preprocessed_occluded, axis=0)
            preds = model.predict(occluded_tensor, verbose=0)[0]
            all_predictions.append(preds)
            patch_positions.append((i, j))

            if patch_count % 20 == 0:
                print(f"  ... processed {patch_count}/{total_patches} patches")

    all_predictions = np.array(all_predictions)  # (num_patches, 9)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    sensitivity_maps = []

    # Generate sensitivity maps for all 9 classes
    for class_idx in range(9):
        print(f"  Processing class {class_idx}: {class_names[class_idx]}...")

        baseline_score = baseline_preds[class_idx]
        sensitivity_map = np.zeros((img_height, img_width))

        # Fill sensitivity map
        for patch_idx, (i, j) in enumerate(patch_positions):
            occluded_score = all_predictions[patch_idx, class_idx]
            sensitivity = baseline_score - occluded_score

            # Fill the patch region
            sensitivity_map[i:i+patch_size, j:j+patch_size] = np.maximum(
                sensitivity_map[i:i+patch_size, j:j+patch_size],
                sensitivity
            )

        # Normalize
        sensitivity_map = np.maximum(sensitivity_map, 0)
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        sensitivity_maps.append(sensitivity_map)

        # Plot
        axes[class_idx].imshow(sensitivity_map, cmap='jet')
        axes[class_idx].axis('off')

        # Create title
        title = f"{class_names[class_idx]}\n{baseline_preds[class_idx]*100:.2f}%"
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
        f"Occlusion Sensitivity (All 9 Classes)\nImage from: {img_source}",
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
        "probabilities": baseline_preds,
        "sensitivity_maps": sensitivity_maps
    }
