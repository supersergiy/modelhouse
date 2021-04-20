import torch
from torch import nn

class Blackouter(nn.Module):
    def __init__(self, fill_value=0):
        super().__init__()
        self.fill_value = 0

    def forward(self, src_img, src_mask, **kwargs):
        result = torch.ones_like(src_img, device=src_img.device)
        result *= self.fill_value
        return result


def create(**kwargs):
    return Blackouter(**kwargs)
