import torch
from torch.utils.data import Dataset
import numpy as np

class SlidingWindowDataset(Dataset):
    def __init__(self, values, labels, window_size=120, stride=5):
        assert len(values) == len(labels)

        self.window_size = window_size
        self.stride = stride

        values = np.asarray(values, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        self.mean = values.mean()
        self.std = values.std() + 1e-8
        values = (values - self.mean) / self.std

        self.windows = []
        self.window_labels = []

        for i in range(0, len(values) - window_size + 1, stride):
            window = values[i:i + window_size]
            label = 1 if labels[i:i + window_size].any() else 0

            self.windows.append(window)
            self.window_labels.append(label)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx]).unsqueeze(-1)
        y = torch.tensor(self.window_labels[idx])
        return x, y
