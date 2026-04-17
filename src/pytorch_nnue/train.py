import torch
from torch import nn
from pytorch_nnue.model import NNUE
import time


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Using {} device".format(device))

model = NNUE().to(device)
start_time = time.time()

X_us = torch.tensor([[10, 500, 40000]], device=device)
X_them = torch.tensor([[15, 600, 35000]], device=device)

pred = model(X_us, X_them)
pred_s = nn.Sigmoid()(pred)
end_time = time.time()
print(f"Predicted value: {pred.item()}, Sigmoid: {pred_s.item()}")
print("Inference time: {} seconds".format(end_time - start_time))   