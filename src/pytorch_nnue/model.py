import torch
from torch import nn
from pytorch_nnue.utils import CReLU

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(768 * 2, 512),
            CReLU(clip_value=255.0),
            nn.Linear(512, 32),
            CReLU(clip_value=255.0),
            nn.Linear(32, 32),
            CReLU(clip_value=255.0),
            nn.Linear(32, 1)
        )
        
        self.feature_transformer = nn.EmbeddingBag(40960, 768, mode='sum')
        
    def forward(self, x_us, x_them):
        x_us = self.feature_transformer(x_us)
        x_them = self.feature_transformer(x_them)
                       
        x = torch.cat([x_us, x_them], dim=1)
        
        x = x.clamp(0.0, 255.0)
        
        logits = self.linear_stack(x)

        return logits