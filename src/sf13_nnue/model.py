import torch
from torch import nn
from sf13_nnue.utils import CReLU

# HYPERPARAMETERS
ACCUMULATOR_SIZE = 256
L1_SIZE = 32
L2_SIZE = 32
CLIP_VALUE = 127.0

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stacks = nn.Sequential(
            nn.Linear(ACCUMULATOR_SIZE * 2, L1_SIZE),
            CReLU(clip_value=CLIP_VALUE),
            nn.Linear(L1_SIZE, L2_SIZE),
            CReLU(clip_value=CLIP_VALUE),
            nn.Linear(L2_SIZE, 1)
        )
        
        self.input = nn.EmbeddingBag(40961, ACCUMULATOR_SIZE, mode='sum', padding_idx=40960)
        
        self.input_bias = nn.Parameter(torch.zeros(ACCUMULATOR_SIZE))
        
    def forward(self, x_us, x_them):
        x_us = self.input(x_us) + self.input_bias
        x_them = self.input(x_them)  + self.input_bias

        x = torch.cat([x_us, x_them], dim=1)
        x = x.clamp(0.0, CLIP_VALUE)
        
        logits = self.layer_stacks(x)

        return logits