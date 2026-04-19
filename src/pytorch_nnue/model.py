import torch
from torch import nn
from pytorch_nnue.utils import CReLU, Lambda

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(768 * 2, 512),
            Lambda(CReLU.apply),
            nn.Linear(512, 32),
            Lambda(CReLU.apply),
            nn.Linear(32, 32),
            Lambda(CReLU.apply),
            nn.Linear(32, 1)
        )
        
        self.feature_transformer = nn.EmbeddingBag(40961, 768, mode='sum', padding_idx=40960)
        
    def forward(self, x_us, x_them):
        x_us = torch.where((x_us == -1) | (x_us >= 40960), 40960, x_us)
        x_them = torch.where((x_them == -1) | (x_them >= 40960), 40960, x_them)
        
        x_us = self.feature_transformer(x_us)  # (N, 768)
        x_them = self.feature_transformer(x_them)  # (N, 768)
        # print(f"x_us after embedding: {x_us.shape}, x_them after embedding: {x_them.shape}")
        # print(x_us)
        # print(x_them)
        x = torch.cat([x_us, x_them], dim=1)
        x = x.clamp(0.0, 255.0)
        
        logits = self.linear_stack(x)

        return logits