"""
Skin Lesion Prediction Module
Standalone module for predicting skin cancer types from images using trained models.

Usage:
    from predict_skin_lesion import predict_image, load_model
    
    model = load_model('best_densenet201.keras')
    top_idx, top_probs = predict_image('path/to/image.jpg', model, top_k=3)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Class names in the order used during training
CLASS_NAMES = [
    'actinic keratosis', 
    'basal cell carcinoma', 
    'dermatofibroma', 
    'melanoma',
    'nevus', 
    'pigmented benign keratosis', 
    'seborrheic keratosis',
    'squamous cell carcinoma', 
    'vascular lesion'
]

# Image dimensions used during training
IMG_H, IMG_W = 75, 100


def load_model(model_path):
    """
    Load a trained Keras model.
    
    Args:
        model_path: Path to the saved .keras model file
        
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    return model


def preprocess_image_for_model(img):
    """
    Preprocess image for DenseNet model prediction.
    
    Args:
        img: PIL.Image object or numpy array
        
    Returns:
        Preprocessed batch ready for model prediction (shape: 1, 75, 100, 3)
    """
    # Convert to RGB if PIL Image
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = img.resize((IMG_W, IMG_H))  # Note: resize expects (width, height)
        img_arr = np.asarray(img)
    else:
        # Assume numpy array
        img_arr = np.array(img)
        if img_arr.shape[0] != IMG_H or img_arr.shape[1] != IMG_W:
            img = Image.fromarray(img_arr.astype('uint8')).resize((IMG_W, IMG_H))
            img_arr = np.asarray(img)
    
    img_arr = img_arr.astype("float32")
    # Use DenseNet preprocessing (same as training)
    img_arr = tf.keras.applications.densenet.preprocess_input(img_arr)
    # Add batch dimension
    batch = np.expand_dims(img_arr, axis=0)
    return batch


def predict_image(path_or_pil, model_obj, top_k=3, show=True):
    """
    Predict skin lesion class for an image.
    
    Args:
        path_or_pil: File path (string) or PIL.Image object
        model_obj: Loaded keras model
        top_k: Number of top predictions to return (default: 3)
        show: Whether to display image and print results (default: True)
    
    Returns:
        tuple: (top_idx, top_probs)
            - top_idx: numpy array of top-k class indices
            - top_probs: numpy array of top-k probabilities
            
    Example:
        >>> model = load_model('best_densenet201.keras')
        >>> top_idx, top_probs = predict_image('lesion.jpg', model, top_k=3)
        >>> print(f"Predicted class: {CLASS_NAMES[top_idx[0]]}")
        >>> print(f"Confidence: {top_probs[0]*100:.2f}%")
    """
    # Extract true label from path if available
    true_label = None
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert("RGB")
        # Try to extract true class from folder name in path
        path_parts = path_or_pil.replace('\\', '/').split('/')
        for part in path_parts:
            if part in CLASS_NAMES:
                true_label = part
                break
    else:
        img = path_or_pil
    
    # Preprocess image
    batch = preprocess_image_for_model(img)
    
    # Get predictions
    preds = model_obj.predict(batch, verbose=0)
    probs = np.squeeze(preds)
    
    # Get top-k predictions
    top_k = min(top_k, len(probs))
    top_idx = probs.argsort()[-top_k:][::-1]
    top_probs = probs[top_idx]
    
    # Display results
    if show:
        plt.figure(figsize=(4, 4))
        if isinstance(img, Image.Image):
            plt.imshow(img)
        else:
            plt.imshow(np.asarray(img).astype("uint8"))
        plt.axis('off')
        
        if true_label:
            plt.title(f"Input Image\nTrue class: {true_label}", fontsize=10, fontweight='bold')
        else:
            plt.title("Input Image", fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print predictions
        if true_label:
            print(f"TRUE LABEL: {true_label}")
            print("-" * 60)
        
        print("PREDICTIONS:")
        for i, (idx, p) in enumerate(zip(top_idx, top_probs)):
            cname = CLASS_NAMES[idx]
            marker = " ✓ CORRECT" if (true_label and cname == true_label and i == 0) else ""
            print(f" {i+1}. {cname:30s}  —  {p*100:5.2f}%{marker}")
        
        # Show if prediction was correct
        if true_label:
            predicted_class = CLASS_NAMES[top_idx[0]]
            if predicted_class == true_label:
                print("\n✓ Prediction is CORRECT!")
            else:
                print(f"\n✗ Prediction is WRONG (predicted '{predicted_class}' but true is '{true_label}')")
    
    return top_idx, top_probs


def predict_batch(image_paths, model_obj, top_k=1):
    """
    Predict skin lesion classes for multiple images.
    
    Args:
        image_paths: List of file paths
        model_obj: Loaded keras model
        top_k: Number of top predictions per image (default: 1)
        
    Returns:
        list: List of (top_idx, top_probs) tuples for each image
    """
    results = []
    for i, path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {path}")
        top_idx, top_probs = predict_image(path, model_obj, top_k=top_k, show=False)
        results.append((top_idx, top_probs))
    return results


def get_class_name(class_idx):
    """
    Get class name from class index.
    
    Args:
        class_idx: Integer class index (0-8)
        
    Returns:
        Class name string
    """
    if 0 <= class_idx < len(CLASS_NAMES):
        return CLASS_NAMES[class_idx]
    else:
        raise ValueError(f"Invalid class index: {class_idx}. Must be between 0 and {len(CLASS_NAMES)-1}")


# Module information
__version__ = '1.0.0'
__author__ = 'Skin Cancer Classification Project'
__all__ = ['predict_image', 'predict_batch', 'load_model', 'get_class_name', 'CLASS_NAMES', 'preprocess_image_for_model']


if __name__ == "__main__":
    # Example usage when running directly
    print("Skin Lesion Prediction Module")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"\nAvailable classes ({len(CLASS_NAMES)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print("\n" + "=" * 60)
    print("\nUsage example:")
    print("  from predict_skin_lesion import predict_image, load_model")
    print("  model = load_model('best_densenet201.keras')")
    print("  top_idx, top_probs = predict_image('lesion.jpg', model, top_k=3)")
