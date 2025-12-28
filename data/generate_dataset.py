import numpy as np
import pandas as pd
import os

np.random.seed(7)

TOTAL_POINTS = 300_000
ANOMALY_RATE = 0.015
SEASONAL_PERIOD = 144

time = np.arange(TOTAL_POINTS)

trend = 0.00004 * time
seasonal = 8 * np.sin(2 * np.pi * time / SEASONAL_PERIOD)
noise = np.random.normal(0, 1.2, TOTAL_POINTS)

values = 50 + trend + seasonal + noise
labels = np.zeros(TOTAL_POINTS)

num_anomalies = int(TOTAL_POINTS * ANOMALY_RATE)

for _ in range(num_anomalies):
    idx = np.random.randint(0, TOTAL_POINTS)

    anomaly_type = np.random.choice(
        ["spike", "level_shift", "variance", "context"]
    )

    if anomaly_type == "spike":
        values[idx] += np.random.uniform(25, 45)
        labels[idx] = 1

    elif anomaly_type == "level_shift":
        length = np.random.randint(80, 300)
        shift = np.random.uniform(12, 20)
        end = min(idx + length, TOTAL_POINTS)
        values[idx:end] += shift
        labels[idx:end] = 1

    elif anomaly_type == "variance":
        length = np.random.randint(80, 250)
        end = min(idx + length, TOTAL_POINTS)
        values[idx:end] += np.random.normal(0, 6, end - idx)
        labels[idx:end] = 1

    elif anomaly_type == "context":
        values[idx] += 12 * np.sin(2 * np.pi * idx / (SEASONAL_PERIOD / 2))
        labels[idx] = 1

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=TOTAL_POINTS, freq="10min"),
    "value": values,
    "label": labels.astype(int)
})

train = df.iloc[:210_000]
val   = df.iloc[210_000:255_000]
test  = df.iloc[255_000:]

os.makedirs("dataset", exist_ok=True)

train.to_csv("dataset/train.csv", index=False)
val.to_csv("dataset/val.csv", index=False)
test.to_csv("dataset/test.csv", index=False)

print("Dataset created successfully")
print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)
