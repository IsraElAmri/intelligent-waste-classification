"""
Two-Phase Fine-Tuning Training Pipeline for Waste Classification.
Aligned with notebooks/waste_classification_resnet50v2.ipynb.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from .model import build_resnet50v2_classifier, unfreeze_for_fine_tuning, DEFAULT_CLASSES

def build_data_generators(train_dir, val_dir=None, img_size=(224, 224), batch_size=16):
    """
    Builds training and validation data generators.
    Validation generator uses rescale-only (no data augmentation).
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 if val_dir is None else 0.0
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2 if val_dir is None else 0.0
    )

    if val_dir is not None:
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        val_gen = val_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

    return train_gen, val_gen

def train_pipeline(data_dir, output_model_path="best_waste_classifier.h5", val_dir=None):
    """
    Two-phase training:
      Phase 1: Classification head warmup (30 epochs, Adam lr=1e-4)
      Phase 2: Fine-tuning top 30 layers (50 epochs, Adam lr=1e-5)
    """
    train_gen, val_gen = build_data_generators(data_dir, val_dir=val_dir, batch_size=16)
    num_classes = len(train_gen.class_indices)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    model, base_model = build_resnet50v2_classifier(num_classes=num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(output_model_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    # Phase 1: Train classification head
    print(">>> Starting Phase 1: Classification Head Warmup (30 epochs, lr=1e-4)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tuning top 30 backbone layers
    print(">>> Starting Phase 2: Backbone Fine-Tuning (50 epochs, lr=1e-5, 30 layers unfrozen)...")
    unfreeze_for_fine_tuning(base_model, num_layers_to_unfreeze=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    print(f">>> Training completed. Model saved to {output_model_path}")
    return model
