import torch
from torch import nn
from sf13_nnue.model import NNUE
from sf13_nnue.data_loader import HalfKPDataset
from sf13_nnue.utils import weight_init, hybrid_loss, AlphaScaler, mse_loss, save_checkpoint
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

def training_loop(dataloader, model, loss_fn, optimizer, scheduler, device, scaler, alpha_scaler: AlphaScaler, mse_fn):
    print("Training...")
    model.train()
    start = time.perf_counter()
    times = []
    for batch, (X, y) in enumerate(dataloader):
        X_us, X_them = X
        score, WDL = y
        
        X_us = X_us.to(device, non_blocking=True)
        X_them = X_them.to(device, non_blocking=True)

        score = score.to(device, non_blocking=True)
        WDL = WDL.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device):
            pred = model(X_us, X_them).squeeze(1)
            loss = loss_fn(pred, score, WDL, alpha=1 - alpha_scaler.get_alpha())
            mse_loss = mse_fn(pred, score)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() 
        alpha_scaler.step()
        
        if batch > 50:
            times.append(time.perf_counter() - start)
        start = time.perf_counter()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f} - mse_loss: {mse_loss.item():>7f} - alpha: {alpha_scaler.get_alpha():.4f}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"

    dataset = HalfKPDataset(batch_size=BATCH_SIZE, shuffle=True, data_dir='D:/Projects/HalfKP Dataset Train')

    dataloader = DataLoader(
            dataset,
            batch_size=None,          
            num_workers=6,
            pin_memory=True,          
            persistent_workers=True,
            prefetch_factor=4,
        )

    scaler = GradScaler()

    model = NNUE().to(device)
    model.apply(weight_init)
    
    model = torch.compile(model) if is_cuda else model

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler1 = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))
    scheduler2 = LinearLR(optimizer, start_factor=0.1, total_iters=5000) 
    scheduler = ChainedScheduler([scheduler2, scheduler1])

    alpha_scaler = AlphaScaler()
    print(alpha_scaler)

    for epoch in range(EPOCHS):
        # Warmup jusqu'à 3 epochs, puis maintien d'un alpha de 0.005 (pondération du BCE)
        if epoch == 0:
            alpha_scaler.set_linear_schedule(initial_alpha=0.0, final_alpha=0.005, total_steps=3 * len(dataloader))
        if epoch == 1:
            alpha_scaler.set_constant_alpha(0.005)

        print(f"Epoch {epoch+1}\n-------------------------------")
        training_loop(dataloader, model, hybrid_loss, optimizer, scheduler, device, scaler, alpha_scaler=alpha_scaler, mse_fn=mse_loss)
        checkpoint_path = f'weights/version2/checkpoint_epoch_{epoch}.pt'
        save_checkpoint(model, optimizer, scheduler, alpha_scaler, epoch, checkpoint_path)        
