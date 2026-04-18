import torch
from torch import nn
from pytorch_nnue.model import NNUE
import time

# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 8192
LR = 0.001

def training_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X_us, X_them = X
        pred = model(X_us, X_them)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  {current:>5d}")

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