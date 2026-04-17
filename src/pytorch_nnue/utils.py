import torch
import torch.nn.functional as F

class CReLU(torch.nn.Module):
    def __init__(self, clip_value=255.0):
        super().__init__()
        self.clip_value = clip_value

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=self.clip_value)
    
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)