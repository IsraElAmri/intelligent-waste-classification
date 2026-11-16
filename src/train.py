"""
Two-Phase Fine-Tuning Training Pipeline for Industrial Waste Classification.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from .model import build_resnet50v2_classifier, unfreeze_for_fine_tuning, DEFAULT_CLASSES

def build_data_generators(data_dir, img_size=(224, 224), batch_size=16):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen

def train_pipeline(data_dir, output_model_path="best_waste_classifier.h5"):
    train_gen, val_gen = build_data_generators(data_dir)
    num_classes = len(train_gen.class_indices)
    
    # Compute balanced class weights to handle dataset skew
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    model, base_model = build_resnet50v2_classifier(num_classes=num_classes)
    
    # Phase 1: Warmup classification head
    print(">>> Starting Phase 1: Training Classification Head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(output_model_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tuning top backbone layers
    print(">>> Starting Phase 2: Two-Phase Fine-Tuning Backbone...")
    unfreeze_for_fine_tuning(base_model, num_layers_to_unfreeze=40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    print(f">>> Training completed! Model saved to {output_model_path}")
    return model
