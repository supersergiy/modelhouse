import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


import os
import pathlib

import artificery
import scalenet
import torchfields

from optimizer import optimize_pre_post_multiscale_ups

def overfit_opt(src, tgt, pred_res_start, src_defects=None, tgt_defects=None, lr=18e-1):

    mse_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'eq', 'value': 0},
             "coarsen_ranges": [ (1, 0), (10, 2)] },
            {'name': 'src',
             "coarsen_ranges": [(1, 0)],
             'fm': 0,
             'binarization': {'strat': 'neq', 'value': 0}
             }
            ],
        'tgt':[
            {'name': 'tgt_defects',
             'binarization': {'strat': 'eq', 'value': 0},
             "coarsen_ranges": [(4, 0)]},
            {'name': 'tgt',
             'fm': 0,
             "coarsen_ranges": [(4, 0)],
             'binarization': {'strat': 'neq', 'value': 0}
             }
        ]
    }
    sm_keys_to_apply = {
       "src": [
           {"name": "src_defects",
            "binarization": {"strat": "eq", "value": 0},
            "coarsen_ranges": [[1, 0], [4, 5]],
            "mask_value": 1.0e-9},
         {"name": "src",
            "fm": 0,
            "binarization": {"strat": "neq", "value": 0}}
       ]
   }


    mips = [4]
    src_small_defects = None
    src_large_defects = None

    if src_defects is not None:
        src_defects = src_defects.squeeze(0)
    else:
        src_defects = torch.zeros_like(src)

    if src_small_defects is not None:
        src_small_defects = src_small_defects.squeeze(0)
    else:
        src_small_defects = torch.zeros_like(src)

    if src_large_defects is not None:
        src_large_defects = src_large_defects.squeeze(0)
    else:
        src_large_defects = torch.zeros_like(src)

    if tgt_defects is not None:
        tgt_defects = tgt_defects.squeeze(0)
    else:
        tgt_defects = torch.zeros_like(src_defects)

    pred_res_opt = optimize_pre_post_multiscale_ups(pred_res_start, src, tgt, mips,
            src_defects=src_defects,
            tgt_defects=tgt_defects,
            src_small_defects=src_defects,
            src_large_defects=src_defects,
            crop=16, bot_mip=4, img_mip=4, max_iter=400,
            sm_keys_to_apply=sm_keys_to_apply,
            mse_keys_to_apply=mse_keys_to_apply,
            sm_val=50e0, lr=lr)
    return pred_res_opt

def create_model(checkpoint_folder, device='cpu'):
    a = artificery.Artificery()

    spec_path = os.path.join(checkpoint_folder, "model_spec.json")
    my_p = a.parse(spec_path)

    checkpoint_path = os.path.join(checkpoint_folder,
            "checkpoint.state.pth.tar")
    if not os.path.isfile(checkpoint_path):
        raise Exception(f"Checkput path {checkpoint_path} no found!")

    load_my_state_dict(my_p,
            torch.load(checkpoint_path,
                map_location=torch.device(device)))

    return my_p


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        this_folder = pathlib.Path(__file__).parent.absolute()
        self.net = create_model(os.path.join(this_folder, 'model'))
        self.ft = True

    def forward(self, src_img, tgt_img, src_agg_field=None, tgt_agg_field=None,
            src_folds=None, tgt_folds=None, **kwargs):

        with torch.no_grad():
            if 'cuda' in str(src_img.device):
                self.net = self.net.cuda(src_img.device)
            else:
                self.net = self.net.cpu()
            print ("MSE pre: ", (src_img - tgt_img).abs().mean())
            if src_agg_field is not None:
                src_agg_field = src_agg_field.field().from_pixels()
                warped_src_img = src_agg_field(src_img)
            else:
                warped_src_img = src_img
            #$if tgt_agg_field is not None:
            #    tgt_agg_field = tgt_agg_field.field().from_pixels()
            #    tgt_img = tgt_agg_field(tgt_img)

            print ("MSE post: ", (warped_src_img - tgt_img).abs().mean())

            net_input = torch.cat((warped_src_img, tgt_img), 1).float()
            pred_res = self.net.forward(x=net_input, level_in=4)
            print ("MSE final: ", (pred_res.field().from_pixels()(warped_src_img) - tgt_img).abs().mean())
            if src_agg_field is not None:
                pred_res = pred_res.field().from_pixels()(src_agg_field).pixels()

        if self.ft == True:
            src = self.net.state['up']['4']['output'][0:4]
            tgt = self.net.state['up']['4']['output'][4:]
            pred_res = overfit_opt(src_img, tgt_img, pred_res.permute(0, 2, 3, 1),
                    src_defects=src_folds, tgt_defects=tgt_folds,
                    lr=3e-1).permute(0, 3, 1, 2)

        return pred_res



def create(**kwargs):
    return Model(**kwargs)
