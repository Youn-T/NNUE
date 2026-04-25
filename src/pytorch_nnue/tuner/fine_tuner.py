import torch
from torch import nn
from pytorch_nnue.model import NNUE
from pytorch_nnue.tuner.tuner_data_loader import HalfKPDatasetTuning
from pytorch_nnue.data_loader import HalfKPDataset
from pytorch_nnue.utils import weight_init, hybrid_loss, AlphaScaler, mse_loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import time
# HYPERPARAMETERS
EPOCHS = 5
BATCH_SIZE = 1024*32
LR = 0.0001/10  # Diviser le taux d'apprentissage par 10 pour le fine-tuning

def training_loop(dataloader, model, loss_fn, optimizer, scheduler, device, scaler, alpha_scaler: AlphaScaler, mse_fn):
    print(f"Training... Batches per epoch: {len(dataloader)}")
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
            # print(pred[:5], score[:5], WDL[:5])
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f} - mse_loss: {mse_loss.item():>7f} - alpha: {alpha_scaler.get_alpha():.4f}")

def load_model(model_path): 
    print("Initialisation du modèle...")
    model = NNUE()

    # 2. Charger les poids de manière robuste (gestion du '_orig_mod.')
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace("_orig_mod.", "")
            clean_state_dict[clean_key] = value
            
        model.load_state_dict(clean_state_dict)
        print("Poids chargés avec succès !")
    except Exception as e:
        print(f"Attention, erreur lors du chargement des poids : {e}")
        print("Le script va continuer avec des poids aléatoires pour le test.")
    return model

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using {} device".format(device))
    is_cuda = device == "cuda"

    dataset = HalfKPDatasetTuning(batch_size=BATCH_SIZE, shuffle=True, data_dir='D:/Projects/HalfKP Dataset Tuning/Base', tuning_data_dir='D:/Projects/HalfKP Dataset Tuning/Lichess')
    # dataset = HalfKPDataset(batch_size=BATCH_SIZE, shuffle=True, data_dir='D:/Projects/HalfKP Dataset Train')

    dataloader = DataLoader(
            dataset,
            batch_size=None,          
            num_workers=6,
            pin_memory=True,          
            persistent_workers=True,
            prefetch_factor=4,
        )

    scaler = GradScaler()

    model = load_model("weights/weights4/model_weights_9.pth")
    model.to(device)
    
    model = torch.compile(model) if is_cuda else model

    optimizer = AdamW(
        [
        {"params": model.feature_transformer.parameters(), "lr": 0.0001},
        {"params": model.linear_stack[0].parameters(), "lr": 0.0001},
        {"params": model.linear_stack[2].parameters(), "lr": 0.00001},
        {"params": model.linear_stack[4].parameters(), "lr": 0.00001},
        {"params": model.linear_stack[6].parameters(), "lr": 0.00001}
    ], lr=LR)
    scheduler1 = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))
    scheduler = scheduler1
    
    alpha_scaler = AlphaScaler()
    print(alpha_scaler)

    for epoch in range(EPOCHS):
        alpha_scaler.set_constant_alpha(0.0025)

        print(f"Epoch {epoch+1}\n-------------------------------")
        training_loop(dataloader, model, hybrid_loss, optimizer, scheduler, device, scaler, alpha_scaler=alpha_scaler, mse_fn=mse_loss)
        torch.save(model.state_dict(), f'weights/weights4_tuned3/model_weights_{epoch}.pth')
        
