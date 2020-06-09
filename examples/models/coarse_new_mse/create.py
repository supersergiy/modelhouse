import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


import os
import pathlib

import artificery

def create_model(checkpoint_folder, device):
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
    def __init__(self, device='cpu'):
        super().__init__()

        this_folder = pathlib.Path(__file__).parent.absolute()
        self.net = create_model(os.path.join(this_folder, 'model'),
                device=device)

    def forward(self, src_img, tgt_img, **kwargs):
        with torch.no_grad():
            if 'cuda' in str(src_img.device):
                self.net = self.net.cuda()
            else:
                self.net = self.net.cpu()


            net_input = torch.cat((src_img, tgt_img), 1).float()
            pred_res = self.net.forward(x=net_input, level_in=7)
        return pred_res



def create(**kwargs):
    return Model(**kwargs)
