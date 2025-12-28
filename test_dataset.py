import pandas as pd
from src.dataset import SlidingWindowDataset

df = pd.read_csv("dataset/train.csv")

ds = SlidingWindowDataset(
    df["value"].values,
    df["label"].values
)

print("Number of windows:", len(ds))

x, y = ds[0]
print("Window shape:", x.shape)
print("Label:", y)
