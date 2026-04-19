import torch
from torch import nn
from pytorch_nnue.model import NNUE
from pytorch_nnue.data_loader import HalfKPDataset
from pytorch_nnue.utils import weight_init, hybrid_loss, halfkp_collate_fn, sanitize_halfkp_indices, get_nstm_indices
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import time
# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 1024*16
LR = 0.001

def training_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
    print("Training...")
    model.train()
    
    start = time.perf_counter()
    times = []
    for batch, (X, king_sq, y) in enumerate(dataloader):
        
        X_us = X
        
        score, WDL = y
        
        # Transfer first, then compute mirrored indices on device to reduce CPU pressure.
        X_us = X_us.to(device, non_blocking=True)
        king_sq = king_sq.to(device, non_blocking=True)
        X_them = get_nstm_indices(X_us, king_sq)
        
        score = score.to(device, non_blocking=True)
        WDL = WDL.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        pred = model(X_us, X_them).squeeze(1)

        loss = loss_fn(pred, score, WDL)
        
        loss.backward()
        optimizer.step()
        scheduler.step() 
        # print(f"Batch {batch+1} - Elapsed time: {time.perf_counter() - start:.4f} ms")
        times.append(time.perf_counter() - start)
        start = time.perf_counter()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  {current:>5d}")
        if batch == 1000:
            print(f"Average batch time: {sum(times) / len(times):.4f} ms -> {1 / ((sum(times) / len(times))):.2f} batches/s -> {((500_000_000 / BATCH_SIZE) * ((sum(times) / len(times))) / 60):.2f} minutes estimated for 500M samples")
        




# X_us = torch.tensor([[10, 500, 40000]], device=device)
# X_them = torch.tensor([[15, 600, 35000]], device=device)

# pred = model(X_us, X_them)
# pred_s = nn.Sigmoid()(pred)
# print(f"Predicted value: {pred.item()}, Sigmoid: {pred_s.item()}")
if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"

    dataset = HalfKPDataset(batch_size=BATCH_SIZE, shuffle=False)

    dataloader = DataLoader(
            dataset,
            batch_size=None,          # Le batching est désormais géré nativement (sans overhead) par le Dataset
            num_workers=6,
            pin_memory=True,          # Extrèmement important pour le transfert rapide vers le GPU
            persistent_workers=True,
            prefetch_factor=4,
        )

    model = NNUE().to(device)
    model.apply(weight_init)


    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        training_loop(dataloader, model, hybrid_loss, optimizer, scheduler, device)