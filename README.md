# Intelligent Industrial Waste Classification System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org/)
[![ResNet50V2](https://img.shields.io/badge/Backbone-ResNet50V2-green)](https://keras.io/api/applications/resnet/)
[![Validation Accuracy](https://img.shields.io/badge/Accuracy-98.23%25-brightgreen)](#-performance-benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep Learning Waste Classification System designed during an AI & Machine Learning Internship at Leoni Wiring Systems (Messadine, Tunisia).**  
> Identifies **8 distinct waste categories** across **6,400+ images** with **98.23% validation accuracy**, utilizing transfer learning, class weighting, and test-time augmentation (TTA).

---

## 📌 Project Overview

Industrial production environments generate heterogeneous waste streams. Efficient recycling and compliance require accurate, real-time sorting. This project develops an end-to-end computer vision pipeline capable of classifying industrial waste with high precision, low false-positive rates, and resilience to lighting/orientation changes.

### 🌟 Key Highlights
- **Architecture**: Transfer learning utilizing **ResNet50V2** pre-trained on ImageNet, fine-tuned with a customized classification head.
- **8 Target Categories**: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`, `battery`, `biological`.
- **Training Strategy**: Two-phase fine-tuning (frozen backbone warmup followed by unfreezing the top 40 residual layers with differential learning rates).
- **Class Imbalance Mitigation**: Skewed class distributions handled with compute-weighted loss weighting and targeted data augmentation.
- **Inference Robustness**: Test-Time Augmentation (TTA) with rotation and flip averaging, coupled with a confidence thresholding gate ($p \ge 0.75$) to route uncertain items for manual review.

---

## 📊 Performance Benchmarks

| Metric | Score | Details |
|---|---|---|
| **Overall Validation Accuracy** | **98.23%** | Evaluated on held-out validation set |
| **Top-1 Accuracy** | **98.23%** | Over 8 waste categories |
| **Dataset Scale** | **6,400+ Images** | Balanced & augmented |
| **Inference Latency** | **~24 ms / image** | GPU accelerated |

### 🎯 Confusion Matrix Summary
```
                cardboard  glass  metal  paper  plastic  trash  battery  biological
cardboard       [ 384       0      1      1       0      0       0         0      ]
glass           [   1     380      0      2       0      0       0         0      ]
metal           [   7       1    373      0       0      1       0         0      ]
paper           [   0       0      0    377       3      0       4         0      ]
plastic         [   0       0      0      8     363      0       1         0      ]
trash           [   0       0      0      0       0    379       2         0      ]
battery         [   0       0      1      0       0      0     381         0      ]
biological      [   0       0      0      0       0      0       0       386      ]
```

---

## 🏗️ Repository Architecture

```
intelligent-waste-classification/
├── notebooks/
│   └── waste_classification_resnet50v2.ipynb  # Interactive experiments & training
├── src/
│   ├── __init__.py
│   ├── model.py                      # ResNet50V2 architecture + head definition
│   ├── train.py                      # Two-phase training pipeline
│   └── predict.py                    # Inference with TTA & confidence gating
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/IsraElAmri/intelligent-waste-classification.git
cd intelligent-waste-classification
pip install -r requirements.txt
```

### 2. Training the Model
```python
from src.train import train_pipeline

# Trains Phase 1 & Phase 2, saving the best weights automatically
model = train_pipeline(data_dir="path/to/dataset", output_model_path="best_waste_classifier.h5")
```

### 3. Inference with Test-Time Augmentation (TTA)
```python
import tensorflow as tf
from src.predict import predict_tta

model = tf.keras.models.load_model("best_waste_classifier.h5")
result = predict_tta(model, image_path="sample_waste.jpg", confidence_threshold=0.80)

print(f"Predicted Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 👩‍💻 Author
**Isra El Amri**  
Mathematical Engineering & Modeling Student at ENSIT  
AI & Machine Learning Enthusiast  
Contact: [elamriisra3@gmail.com](mailto:elamriisra3@gmail.com)
