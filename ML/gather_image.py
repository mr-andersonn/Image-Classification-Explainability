import os
import random
import numpy as np
from PIL import Image


def get_random_image_path(dataset_folder, class_names=None):
    """
    Get a random image path from the fish dataset.

    Parameters:
    -----------
    dataset_folder : str
        Path to the Fish_Dataset folder
    class_names : list, optional
        List of class names to sample from. If None, uses all classes.

    Returns:
    --------
    str: Path to a random image file
    """
    if class_names is None:
        # Get all subdirectories that don't end with GT
        class_names = [d for d in os.listdir(dataset_folder)
                      if os.path.isdir(os.path.join(dataset_folder, d))
                      and not d.endswith('GT')
                      and not d.endswith('.m')
                      and not d.endswith('.txt')]

    # Pick a random class
    random_class = random.choice(class_names)

    # Build path to the nested folder (e.g., "Black Sea Sprat/Black Sea Sprat")
    class_folder = os.path.join(dataset_folder, random_class, random_class)

    # Check if the nested folder exists
    if not os.path.isdir(class_folder):
        raise FileNotFoundError(f"Expected nested folder not found: {class_folder}")

    # Get all image files
    image_files = [f for f in os.listdir(class_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        raise FileNotFoundError(f"No images found in {class_folder}")

    # Pick a random image
    random_image = random.choice(image_files)

    # Return full path
    return os.path.join(class_folder, random_image)


def find_misclassified_image(model, class_names, dataset_folder, max_attempts=1000):
    """
    Find a random image where the model prediction is wrong.

    Parameters:
    -----------
    model : tf.keras.Model
        The trained model
    class_names : list
        List of class names matching the dataset folder structure
    dataset_folder : str
        Path to the Fish_Dataset folder
    max_attempts : int
        Maximum number of images to try before giving up

    Returns:
    --------
    dict or None: {
        "img_path": str,
        "true_class": str,
        "true_class_idx": int,
        "pred_class": str,
        "pred_class_idx": int,
        "confidence": float,
        "attempts": int
    }
    Returns None if no misclassification found after max_attempts.
    """
    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        # Get a random image
        img_path = get_random_image_path(dataset_folder, class_names)

        # Extract true class from path
        # Path format: .../Fish_Dataset/Class Name/Class Name/00001.png
        path_parts = img_path.split(os.sep)
        true_class_name = path_parts[-3]  # Two levels up from filename

        # Get the true class index
        try:
            true_class_idx = class_names.index(true_class_name)
        except ValueError:
            print(f"Warning: Class '{true_class_name}' not in class_names list, skipping...")
            continue

        # Load and preprocess image
        # Note: No preprocessing applied - assumes model has built-in rescaling layer
        # (e.g., MobileNetV3Large_Improved.keras has a Rescaling layer at the start)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype("float32")
        img_tensor = np.expand_dims(arr, axis=0)

        # Predict
        preds = model.predict(img_tensor, verbose=0)[0]
        pred_class_idx = int(np.argmax(preds))

        # Check if prediction is wrong
        if pred_class_idx != true_class_idx:
            print(f"✓ Found misclassification after {attempts} attempts!")
            print(f"  True: {true_class_name} (class {true_class_idx})")
            print(f"  Predicted: {class_names[pred_class_idx]} (class {pred_class_idx})")
            print(f"  Confidence: {preds[pred_class_idx]*100:.2f}%")
            print(f"  Path: {img_path}")

            return {
                "img_path": img_path,
                "true_class": true_class_name,
                "true_class_idx": true_class_idx,
                "pred_class": class_names[pred_class_idx],
                "pred_class_idx": pred_class_idx,
                "confidence": float(preds[pred_class_idx]),
                "attempts": attempts
            }

        # Progress update every 50 attempts
        if attempts % 50 == 0:
            print(f"  ... tried {attempts} images, still searching...")

    print(f"✗ Could not find a misclassification after {max_attempts} attempts")
    print("  (Your model might be very accurate!)")
    return None
