"""
Inference pipeline with Test-Time Augmentation (TTA) and Confidence Thresholding.
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from .model import DEFAULT_CLASSES

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB').resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def predict_single(model, image_path, classes=DEFAULT_CLASSES, confidence_threshold=0.75):
    """
    Standard single-pass prediction.
    """
    img_arr = preprocess_image(image_path)
    batch = np.expand_dims(img_arr, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    best_idx = np.argmax(probs)
    confidence = float(probs[best_idx])
    
    label = classes[best_idx] if confidence >= confidence_threshold else "Uncertain / Manual Inspection Needed"
    return {
        "class": label,
        "confidence": confidence,
        "raw_probabilities": {cls: float(p) for cls, p in zip(classes, probs)}
    }

def predict_tta(model, image_path, classes=DEFAULT_CLASSES, confidence_threshold=0.75):
    """
    Test-Time Augmentation (TTA): evaluates the image across original,
    horizontal flip, and small rotations, then averages the probability distribution.
    """
    base_img = Image.open(image_path).convert('RGB').resize((224, 224))
    
    variants = [
        base_img,
        base_img.transpose(Image.FLIP_LEFT_RIGHT),
        base_img.rotate(5),
        base_img.rotate(-5)
    ]
    
    batch = np.array([np.array(v, dtype=np.float32) / 255.0 for v in variants])
    predictions = model.predict(batch, verbose=0)
    mean_probs = np.mean(predictions, axis=0)
    
    best_idx = np.argmax(mean_probs)
    confidence = float(mean_probs[best_idx])
    
    label = classes[best_idx] if confidence >= confidence_threshold else "Uncertain / Low Confidence"
    return {
        "class": label,
        "confidence": confidence,
        "raw_probabilities": {cls: float(p) for cls, p in zip(classes, mean_probs)}
    }
