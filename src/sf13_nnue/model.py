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
        
        self.input = nn.EmbeddingBag(41024, ACCUMULATOR_SIZE, mode='sum')
        
        self.input_bias = nn.Parameter(torch.zeros(ACCUMULATOR_SIZE))
        
    def forward(self,us, them, w_idx, b_idx, w_offsets, b_offsets):

        
        x_w = self.input(w_idx, w_offsets) + self.input_bias
        x_b = self.input(b_idx, b_offsets) + self.input_bias

        x = (us * torch.cat([x_w, x_b], dim=1)) + (them * torch.cat([x_b, x_w], dim=1))
        x = x.clamp(0.0, CLIP_VALUE)
        
        logits = self.layer_stacks(x)

        return logits