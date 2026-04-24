import torch
from torch import nn
from pytorch_nnue.model import NNUE
from pytorch_nnue.data_loader import HalfKPDataset
from pytorch_nnue.utils import weight_init, mse_loss, halfkp_collate_fn, sanitize_halfkp_indices, get_nstm_indices, remove_dicte_keys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import time
# HYPERPARAMETERS
BATCH_SIZE = 1024*32

def testing_loop(dataloader, model, loss_fn, device):
    model.eval()
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X_us, X_them = X
        
        score, WDL = y
        
        X_us = X_us.to(device, non_blocking=True)
        X_them = X_them.to(device, non_blocking=True)
        
        score = score.to(device, non_blocking=True)
        WDL = WDL.to(device, non_blocking=True)

        with autocast(device_type=device):
            pred = model(X_us, X_them).squeeze(1)

            loss = loss_fn(pred, score)
        losses.append(loss.item())

        if batch % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")
    avg_loss = sum(losses) / len(losses)
    print(f"Average loss for epoch: {avg_loss:.6f}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"

    dataset = HalfKPDataset(batch_size=BATCH_SIZE, shuffle=True, data_dir='D:/Projects/HalfKP Dataset Test')

    dataloader = DataLoader(
            dataset,
            batch_size=None,      
            num_workers=6,
            pin_memory=True,         
            persistent_workers=True,
            prefetch_factor=4,
        )


    model = NNUE().to(device)
    
    state_dict = torch.load('weights2/model_weights_9.pth')
    state_dict = remove_dicte_keys(state_dict)
    
    model.load_state_dict(state_dict)
    # model.apply(weight_init)
    model = torch.compile(model) if is_cuda else model


    testing_loop(dataloader, model, mse_loss, device)
        
