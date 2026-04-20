import torch
from torch import nn
from pytorch_nnue.model import NNUE
from pytorch_nnue.data_loader import HalfKPDataset
from pytorch_nnue.utils import weight_init, hybrid_loss, halfkp_collate_fn, sanitize_halfkp_indices, get_nstm_indices
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import time
# HYPERPARAMETERS
EPOCHS = 10
BATCH_SIZE = 1024*32
LR = 0.0001

def training_loop(dataloader, model, loss_fn, optimizer, scheduler, device, scaler, alpha=1.0):
    print("Training...")
    model.train()
    
    start = time.perf_counter()
    times = []
    for batch, (X, y) in enumerate(dataloader):
        # print(scheduler.get_last_lr())
        X_us, X_them = X
        
        score, WDL = y
        
        # Transfer first, then compute mirrored indices on device to reduce CPU pressure.
        X_us = X_us.to(device, non_blocking=True)
        X_them = X_them.to(device, non_blocking=True)
        # king_sq = king_sq.to(device, non_blocking=True)
        # X_them = get_nstm_indices(X_us, king_sq)
        
        score = score.to(device, non_blocking=True)
        WDL = WDL.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device):
            pred = model(X_us, X_them).squeeze(1)
            # print(pred[:5], score[:5], WDL[:5])
            loss = loss_fn(pred, score, WDL, alpha=alpha, mse_factor=100.0)#alpha)
        
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() 

        # print(f"Batch {batch+1} - Elapsed time: {time.perf_counter() - start:.4f} ms")
        if batch > 50:
            times.append(time.perf_counter() - start)
        start = time.perf_counter()
        if batch % 100 == 0:
            # print(pred[:5], score[:5], WDL[:5])
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}")
        # if batch == 1000:
        #     print(f"Average batch time: {sum(times) / len(times):.4f} ms -> {1 / ((sum(times) / len(times))):.2f} batches/s -> {((500_000_000 / BATCH_SIZE) * ((sum(times) / len(times))) / 60):.2f} minutes estimated for 500M samples")
        




# X_us = torch.tensor([[10, 500, 40000]], device=device)
# X_them = torch.tensor([[15, 600, 35000]], device=device)

# pred = model(X_us, X_them)
# pred_s = nn.Sigmoid()(pred)
# print(f"Predicted value: {pred.item()}, Sigmoid: {pred_s.item()}")
if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"

    dataset = HalfKPDataset(batch_size=BATCH_SIZE, shuffle=True, data_dir='D:/Projects/HalfKP Dataset Train')

    dataloader = DataLoader(
            dataset,
            batch_size=None,          # Le batching est désormais géré nativement (sans overhead) par le Dataset
            num_workers=6,
            pin_memory=True,          # Extrèmement important pour le transfert rapide vers le GPU
            persistent_workers=True,
            prefetch_factor=4,
        )

    scaler = GradScaler()

    model = NNUE().to(device)
    model.apply(weight_init)
    model = torch.compile(model) if is_cuda else model

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler1 = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))#, eta_min=LR/100)  # Décay sur toute la durée de l'entraînement
    scheduler2 = LinearLR(optimizer, start_factor=0.1, total_iters=5000)  # Warmup de 10 itérations
    scheduler = ChainedScheduler([scheduler2, scheduler1])

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        alpha = 1.0 if epoch < 3 else 0.8 if epoch < 8 else 0.5  # Diminution progressive de l'importance du score dans la loss
        training_loop(dataloader, model, hybrid_loss, optimizer, scheduler, device, scaler, alpha=alpha)
        torch.save(model.state_dict(), f'weights/model_weights_{epoch}.pth')