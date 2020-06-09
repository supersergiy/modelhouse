import torch
import torch.nn as nn

from blockmatch import block_match


class Model(nn.Module):
    def __init__(self, tile_size=64, tile_step=32, max_disp=32, r_delta=1.1):
        super().__init__()
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.max_disp = max_disp
        self.r_delta = r_delta

    def __getitem__(self, index):
        return None

    def forward(self, src_img, tgt_img, **kwargs):
        with torch.no_grad():
            pred_res = block_match(src_img, tgt_img, tile_size=self.tile_size,
                                   tile_step=self.tile_step, max_disp=self.max_disp,
                                   min_overlap_px=500, filler=0, r_delta=self.r_delta)
        return pred_res.permute(0, 3, 1, 2)



def create(**kwargs):
    return Model(**kwargs)
