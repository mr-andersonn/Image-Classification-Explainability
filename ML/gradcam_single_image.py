def visualize_gradcam(
    img_path, 
    model, 
    target_layer_name="mobilenetv2_1.00_224",
    class_index=None,
    alpha=0.45,
    save_path=None
):
    """
    Create Grad-CAM visualization for a single image.
    
    Parameters:
    -----------
    img_path : str
        Path to the input image
    model : tf.keras.Model
        The trained model
    target_layer_name : str
        Name of the layer to use for Grad-CAM (default: "mobilenetv2_1.00_224")
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
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    img_tensor = np.expand_dims(arr, axis=0)
    
    # Get the target layer
    mobilenet_layer = model.get_layer(target_layer_name)
    subsequent_layers = model.layers[model.layers.index(mobilenet_layer) + 1:]
    
    # Compute Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs = mobilenet_layer(img_tensor, training=False)
        tape.watch(conv_outputs)
        
        x = conv_outputs
        for layer in subsequent_layers:
            x = layer(x, training=False)
        predictions = x
        
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        
        class_score = predictions[:, class_index]
    
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    # Create overlay
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size)
    )
    
    import matplotlib
    cmap = matplotlib.colormaps['jet']
    colored = cmap(heatmap_resized / 255.0)
    colored = (colored[..., :3] * 255).astype(np.uint8)
    
    overlay = (
        np.array(img).astype(np.float32) * (1 - alpha) + 
        colored.astype(np.float32) * alpha
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay_img)
    axes[2].set_title(f'Overlay (Class {class_index})', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return img, heatmap, overlay_img, class_index
