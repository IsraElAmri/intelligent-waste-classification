"""
Model architecture for Waste Classification using ResNet50V2.
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential

DEFAULT_CLASSES = [
    'cardboard', 'glass', 'metal', 'paper', 
    'plastic', 'trash', 'battery', 'biological'
]

def build_resnet50v2_classifier(num_classes=8, input_shape=(224, 224, 3), dropout_rate=0.4, l2_factor=1e-4):
    """
    Constructs the two-phase transfer learning model with ResNet50V2 backbone.
    """
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_factor)),
        Dropout(dropout_rate),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_factor)),
        Dropout(dropout_rate / 2),
        Dense(num_classes, activation='softmax', dtype='float32')
    ], name="Waste_Classifier_ResNet50V2")

    return model, base_model

def unfreeze_for_fine_tuning(base_model, num_layers_to_unfreeze=30):
    """
    Unfreezes the top N layers of the backbone for Stage 2 fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    return base_model
