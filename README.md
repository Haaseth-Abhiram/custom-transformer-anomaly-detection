# Transformer-Based Time Series Anomaly Detection

## Overview
This project implements a **Transformer-based anomaly detection system** for univariate time-series data.
It detects abnormal patterns such as spikes, level shifts, and contextual anomalies using **self-attention**
and **sliding window sequences**.

The project is designed as an **end-to-end ML pipeline** covering data generation, preprocessing,
model training, and evaluation.

---

## Problem Statement
Traditional anomaly detection techniques struggle with:
- Long-range temporal dependencies
- Contextual anomalies
- Highly imbalanced datasets

This project addresses these challenges using a **Transformer encoder**, which captures global
temporal relationships via self-attention.

---

## Dataset
- Synthetic univariate time-series data
- Includes trend, seasonality, and noise
- Anomaly types:
  - Point anomalies (spikes)
  - Level shifts
  - Variance changes
  - Contextual anomalies
- Strong class imbalance (realistic scenario)

Data is split into:
- Train
- Validation
- Test

**Note:** Large datasets and trained model files are excluded from GitHub and can be regenerated.

---

## Data Preparation
- Time series is converted into overlapping **sliding windows**
- A window is labeled anomalous if **any timestep inside it is anomalous**
- This enables detection of **collective anomalies**

---

## Model Architecture
- Input projection layer
- Positional Encoding
- Transformer Encoder (multi-head self-attention)
- Global average pooling
- Binary classification head

The model outputs a binary anomaly score per window.

---

## Training Strategy
- Binary classification using **BCEWithLogitsLoss**
- Class imbalance handled using **positive class weighting**
- Optimized using Adam optimizer
- Validation-based monitoring during training

---

## Evaluation Metrics
Accuracy is avoided due to class imbalance.

Metrics used:
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Project Structure
```
custom_transformer_anomaly/
│
├── data/
│   └── generate_dataset.py
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── evaluate.py
├── README.md
└── .gitignore
```

---

## How to Run

### Generate Dataset
```
python data/generate_dataset.py
```

### Train Model
```
python -m src.train
```

### Evaluate Model
```
python evaluate.py
```

---

## Key Learnings
- Transformers capture long-term dependencies better than RNNs
- Sliding-window labeling is critical for anomaly detection
- Precision and recall are more meaningful than accuracy
- Clean project structure improves reproducibility

---

## Future Improvements
- Threshold tuning using ROC / PR curves
- Multivariate time-series support
- Streaming / online anomaly detection
- Attention-based explainability

---

## Author
Developed as a **job-oriented machine learning project** focused on real-world anomaly detection.
