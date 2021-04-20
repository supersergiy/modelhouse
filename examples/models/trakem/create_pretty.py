import torch

import numpy as np
from pytrakem import TrakEM2, normalize

class Model(torch.nn.Module):
    def __init__(self, memory_limit=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, src_img, tgt_img, **kwargs):
        t2 = TrakEM2(**self.kwargs)
        torch_stack = torch.stack([src_img.squeeze(), tgt_img.squeeze()], 0)
        np_stack = normalize(torch_stack.cpu().detach().numpy())
        disp = t2.align(np_stack, fixed_section_id=0, return_pytorch=True)
        return (disp[1:]).pixels()

def create(**kwargs):
    return Model(**kwargs)
