import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.dataset import SlidingWindowDataset
from src.model import TransformerAnomalyDetector

# ---------------- CONFIG ----------------
WINDOW_SIZE = 120
STRIDE = 5
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

# Load test data
df = pd.read_csv("dataset/test.csv")

test_ds = SlidingWindowDataset(
    df["value"].values,
    df["label"].values,
    window_size=WINDOW_SIZE,
    stride=STRIDE
)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Load model
model = TransformerAnomalyDetector().to(DEVICE)
model.load_state_dict(torch.load("transformer_anomaly_model.pt", map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = (torch.sigmoid(logits) > 0.5).int()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary", zero_division=0
)

cm = confusion_matrix(all_labels, all_preds)

print("=== TEST SET RESULTS ===")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("\nConfusion Matrix:")
print(cm)
