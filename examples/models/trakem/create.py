import torch

import numpy as np
from pytrakem import TrakEM2

class Model(torch.nn.Module):
    def __init__(self, memory_limit=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, src_img, tgt_img, **kwargs):
        t2 = TrakEM2(**self.kwargs)
        if isinstance(src_img, torch.Tensor):
            torch_stack = torch.stack([src_img.squeeze(), tgt_img.squeeze()], 0)
            torch_stack -= torch_stack.min()
            torch_stack /= torch_stack.max()
            np_stack = torch_stack.cpu().detach().numpy()
        else:
            np_stack = np.stack([src_img, tgt_img], 0)
        np_stack =  (np_stack * 255.0).astype(np.uint8)
        disp = t2.align(np_stack, fixed_section_id=0,
                return_pytorch=True)
        return (~(disp[1:])).pixels()



def create(**kwargs):
    return Model(**kwargs)
