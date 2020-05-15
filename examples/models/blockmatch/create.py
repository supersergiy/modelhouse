import torch
import torch.nn as nn

from blockmatch import block_match
from optimizer import optimize
from residuals import res_warp_img, combine_residuals


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
        if pred_res.shape[1] == 2:
            pred_res = pred_res.permute(0, 2, 3, 1)

        if src_img.var() > 1e-4:
            #pred_res = optimize(src_img, tgt_img, pred_res, torch.zeros_like(src_img),
            #                        torch.zeros_like(tgt_img), max_iter=40)
            pass
        else:
            print ("skipping fucking shit")
        '''warped_tgt = res_warp_img(src_img, pred_res, is_pix_res=True)
        with torch.no_grad():
            refinement_res = block_match(warped_tgt, tgt_img, tile_size=self.tile_size,
                                   tile_step=self.tile_step, max_disp=self.max_disp,
                                   min_overlap_px=400, filler=0)
        if src_img.var() > 1e-4:
            #refinement_res = optimize(warped_tgt, tgt_img, refinement_res, torch.zeros_like(src_img),
            #                        torch.zeros_like(tgt_img), max_iter=200)
            pass
        final_res = combine_residuals(pred_res, refinement_res, is_pix_res=True)'''

        final_res = filter_black_field(pred_res, tgt_img, 0.0)
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
