import torch
from src.model import TransformerAnomalyDetector

model = TransformerAnomalyDetector()

dummy_input = torch.randn(8, 120, 1)
output = model(dummy_input)

print("Output shape:", output.shape)
