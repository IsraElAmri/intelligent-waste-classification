# Waste Classification with ResNet50V2

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-98.23%25-brightgreen)](notebooks/waste_classification_resnet50v2.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An 8-class image classification pipeline using transfer learning with ResNet50V2 to categorize waste streams. Developed during an AI and machine learning internship at Leoni Wiring Systems (IT Department, Messadine, Tunisia).

## Dataset and Scope

- **Dataset**: Derived from the Kaggle Garbage Classification dataset (8 classes used, glass types merged, clothes and shoes excluded); exact upstream preparation not documented.
- **Dataset Confirmation**: Confirmed directly from the executed notebook (`notebooks/waste_classification_resnet50v2.ipynb` Cell 1, 2, and 6), which processes the local folder `Garbage_classification` with 8 categories (`battery`, `biological`, `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
- **Split & Sample Counts**: Tied directly to notebook Cell 6 generator outputs:
  - 5,120 training images (`Found 5120 images belonging to 8 classes`)
  - 2,771 validation images (`Found 2771 images belonging to 8 classes`)
  - 7,891 total images evaluated.
  - No separate held-out test split was used; reported metrics reflect the validation split used for checkpoint selection.
- **Class Balance**: Classes are moderately imbalanced (validation support ranges from 288 in battery/biological/trash to 386 in cardboard). Balanced class weighting was applied during training.
- **Model Checkpoints**: Trained weights (`.h5` / `.keras`) exceed GitHub storage limits and are gitignored. The notebook contains the complete training execution and outputs. The pipeline in `src/train.py` is aligned with the notebook parameters but was not retrained separately.

## Architecture and Training

1. **Backbone**: ResNet50V2 pretrained on ImageNet.
2. **Classification Head**: GlobalAveragePooling2D, Batch Normalization, Dropout(0.5), Dense(512, ReLU, L2=1e-3), Batch Normalization, Dropout(0.3), Dense(256, ReLU, L2=1e-3), Batch Normalization, Dropout(0.2), Dense(8, Softmax).
3. **Phase 1 (Warmup)**: Frozen backbone, Adam optimizer ($lr = 10^{-4}$), 30 epochs.
4. **Phase 2 (Fine-Tuning)**: Top 30 backbone layers unfrozen, Adam optimizer ($lr = 10^{-5}$), 50 epochs.
5. **Data Augmentation**: Training generator applies rotation (30°), width/height shifts (0.2), zoom (0.2), shear (0.15), and horizontal flips. Validation generator uses rescale-only normalization.

## Validation Results

Evaluated on the 2,771-sample validation split:

- **Validation Accuracy**: 98.23% (2,722 / 2,771 correct).
- **Test-Time Augmentation (TTA)**: 98.27% (marginal gain over standard inference).

### Per-Class Validation Breakdown

| Category | Validation Support | Correct Predictions | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| Cardboard | 386 | 384 | 99.5% | 0.9796 | 0.9948 | 0.9871 |
| Glass | 383 | 380 | 99.2% | 0.9948 | 0.9922 | 0.9935 |
| Metal | 382 | 373 | 97.6% | 0.9973 | 0.9764 | 0.9868 |
| Paper | 384 | 377 | 98.2% | 0.9593 | 0.9818 | 0.9704 |
| Plastic | 372 | 363 | 97.6% | 0.9680 | 0.9758 | 0.9719 |
| Trash | 288 | 285 | 99.0% | 0.9828 | 0.9896 | 0.9862 |
| Battery | 288 | 274 | 95.1% | 0.9821 | 0.9514 | 0.9665 |
| Biological | 288 | 286 | 99.3% | 1.0000 | 0.9931 | 0.9965 |

### Error Analysis (Notebook Cell 28)

The model's weakest category is **battery** (274/288 correct, 95.14% recall), with primary misclassifications distributed into trash and metal. The two most frequent individual off-diagonal confusions across the entire validation set are:
1. `plastic → paper`: 8 misclassifications
2. `metal → cardboard`: 7 misclassifications

## Repository Structure

```
intelligent-waste-classification/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── notebooks/
│   └── waste_classification_resnet50v2.ipynb
└── src/
    ├── dataset.py
    ├── model.py
    ├── predict.py
    └── train.py
```

## Getting Started

```bash
pip install -r requirements.txt

# Run the training pipeline
python src/train.py

# Run prediction on a single image
python src/predict.py --image path/to/sample.jpg
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
