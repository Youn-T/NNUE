import torch
from torch import nn
from sf13_nnue.model import NNUE
# from sf13_nnue.data_loader import HalfKPDataset
from sf13_nnue.utils import weight_init, hybrid_loss, AlphaScaler, mse_loss, save_checkpoint, load_checkpoint, make_data_loaders
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import sf13_nnue.nnue_dataset as nnue_dataset 
import time
import data_loader
import features as M
# HYPERPARAMETERS
EPOCHS = 10
BATCH_SIZE = 1024*8
LR = 0.0001

def training_loop(dataloader, model, loss_fn, optimizer, scheduler, device, scaler, alpha_scaler: AlphaScaler, mse_fn):
    print("Training...")
    model.train()
    start = time.perf_counter()
    times = []
    for batch, (us, them, X_w, X_b, WDL, score) in enumerate(dataloader):
        X_w = X_w.to(device=device, non_blocking=True)
        X_b = X_b.to(device=device, non_blocking=True)
        us = us.to(device=device, non_blocking=True)
        them = them.to(device=device, non_blocking=True)


        score = score.to(device, non_blocking=True)
        WDL = WDL.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device):
            pred = model(us, them, X_w, X_b)
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
            # if batch % 10_000_000:
            #     checkpoint_path = f'weights/version3/checkpoint_epoch_{epoch}_{batch}.pt'
            #     save_checkpoint(model, optimizer, scheduler, alpha_scaler, epoch, checkpoint_path, scaler)   
            
            loss = loss.item()
            print(f"loss: {loss:>7f} - mse_loss: {mse_loss.item():>7f} - alpha: {alpha_scaler.get_alpha():.4f}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"


    # train_infinite = nnue_dataset.SparseBatchDataset('HalfKP', 
    #                                                  "D:/Projects/NNUE_SF_13/T60T70wIsRightFarseer.binpack",
    #                                                  BATCH_SIZE,
    #                                                  num_workers=6,
    #                                                  filtered=True,
    #                                                  random_fen_skipping=True,
    #                                                  cyclic=True,
    #                                                  device='cpu')
    # dataloader = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (500_000_000 + BATCH_SIZE - 1) // BATCH_SIZE), batch_size=None, batch_sampler=None)
    halfkp = M.get_feature_set_from_name("HalfKP")
    
    dataloader = make_data_loaders(
        ["D:/Projects/NNUE_SF_13/T60T70wIsRightFarseer.binpack"],
        feature_set=halfkp,
        num_workers=6,
        batch_size=1024 * 8,
        config= data_loader.DataloaderSkipConfig(
            filtered=False,
            random_fen_skipping=0,
            wld_filtered=False,
            early_fen_skipping=False
        ),
        epoch_size=500_000_000,
    )
    scaler = GradScaler()

    model = NNUE().to(device)
    model.apply(weight_init)
    

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler1 = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))
    scheduler2 = LinearLR(optimizer, start_factor=0.1, total_iters=5000) 
    scheduler = ChainedScheduler([scheduler2, scheduler1])

    alpha_scaler = AlphaScaler()
    start_epoch = 0
    
    # Comment this to fully restart training, or set to a checkpoint path to resume from a specific epoch
    # checkpoint_path = "weights/version3/checkpoint_epoch_2.pt"
    # start_epoch, alpha_scaler = load_checkpoint(checkpoint_path, model, optimizer, scheduler, alpha_scaler, scaler)

    model = torch.compile(model) if is_cuda else model


    for epoch in range(start_epoch, EPOCHS):
        # Warmup jusqu'à 3 epochs, puis maintien d'un alpha de 0.005 (pondération du BCE)
        if epoch == 0:
            alpha_scaler.set_linear_schedule(initial_alpha=0.0, final_alpha=0.05, total_steps=3 * len(dataloader))
        if epoch == 3:
            alpha_scaler.set_constant_alpha(0.05)

        print(f"Epoch {epoch+1}\n-------------------------------")
        training_loop(dataloader, model, hybrid_loss, optimizer, scheduler, device, scaler, alpha_scaler=alpha_scaler, mse_fn=mse_loss)
        checkpoint_path = f'weights/version4/checkpoint_epoch_{epoch}.pt'
        save_checkpoint(model, optimizer, scheduler, alpha_scaler, epoch, checkpoint_path, scaler)     
