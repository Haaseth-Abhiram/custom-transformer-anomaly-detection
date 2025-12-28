# Transformer-Based Time Series Anomaly Detection

## Overview
This project implements a custom Transformer-based anomaly detector for univariate time-series data. 
The system is designed to detect point, contextual, and collective anomalies using sliding window sequences.

## Dataset
- Synthetic time-series data with trend and seasonality
- Multiple anomaly types spikes, level shifts, variance changes
- Strong class imbalance (realistic scenario)
- Train  Validation  Test split

## Approach
1. Generate large-scale time-series data
2. Convert data into sliding windows
3. Train a Transformer Encoder with positional encoding
4. Use class-weighted BCE loss to handle imbalance
5. Evaluate using Precision, Recall, and F1-score

## Model Architecture
- Input projection layer
- Positional encoding
- Transformer Encoder (multi-head self-attention)
- Global average pooling
- Binary classification head

## Evaluation Metrics
Due to class imbalance, accuracy is avoided.
Metrics used
- Precision
- Recall
- F1 Score
- Confusion Matrix

## How to Run
```bash
python -m src.train
python evaluate.py
