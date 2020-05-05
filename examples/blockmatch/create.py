import torch
import torch.nn as nn

from blockmatch import block_match
from optimizer import optimize
from residuals import res_warp_img, combine_residuals

class Model(nn.Module):
    def __init__(self, tile_size=64, tile_step=32, max_disp=64):
        super().__init__()
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.max_disp = max_disp

    def __getitem__(self, index):
        return None

    def forward(self, src, tgt, in_field=None, plastic_mask=None, mip_in=6,
                encodings=False, **kwargs):
        with torch.no_grad():
            pred_res = block_match(src, tgt, tile_size=self.tile_size,
                                   tile_step=self.tile_step, max_disp=self.max_disp,
                                   min_overlap_px=500)
        if pred_res.shape[1] == 2:
            pred_res = pred_res.permute(0, 2, 3, 1)

        if src.var() > 1e-4:
            pred_res = optimize(src, tgt, pred_res, torch.zeros_like(src),
                                    torch.zeros_like(tgt), max_iter=40)
            pass
        else:
            print ("skipping fucking shit")
        warped_tgt = res_warp_img(src, pred_res, is_pix_res=True)
        with torch.no_grad():
            refinement_res = block_match(warped_tgt, tgt, tile_size=self.tile_size,
                                   tile_step=self.tile_step, max_disp=self.max_disp,
                                   min_overlap_px=400, filler=0)
        if src.var() > 1e-4:
            refinement_res = optimize(warped_tgt, tgt, refinement_res, torch.zeros_like(src),
                                    torch.zeros_like(tgt), max_iter=200)
            pass
        final_res = combine_residuals(pred_res, refinement_res, is_pix_res=True)

        final_res = filter_black_field(final_res, tgt, 0.05)
        final_res = final_res * 2 / src.shape[-2]
        return final_res

def filter_black_field(field, img, black_threshold=0, permute=True):
    if permute:
        field = field.permute(0, 3, 1, 2)

    black_mask = (img.abs() < black_threshold).squeeze()
    field[..., black_mask] = 0
    if permute:
        field = field.permute(0, 2, 3, 1)
    return field

def create(**kwargs):
    return Model(**kwargs)
