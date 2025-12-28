import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.dataset import SlidingWindowDataset
from src.model import TransformerAnomalyDetector

# ---------------- CONFIG ----------------
WINDOW_SIZE = 120
STRIDE = 5
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

def load_data(split):
    df = pd.read_csv(f"dataset/{split}.csv")
    return SlidingWindowDataset(
        df["value"].values,
        df["label"].values,
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )

train_ds = load_data("train")
val_ds   = load_data("val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = TransformerAnomalyDetector().to(DEVICE)

# Handle class imbalance
labels = torch.tensor(train_ds.window_labels)
pos_weight = (labels == 0).sum() / (labels == 1).sum()

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    p, r, f, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return p, r, f

# ---------------- TRAIN ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    precision, recall, f1 = evaluate(model, val_loader)

    print(
        f"Epoch {epoch} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

# Save model
torch.save(model.state_dict(), "transformer_anomaly_model.pt")
print("Model saved as transformer_anomaly_model.pt")
