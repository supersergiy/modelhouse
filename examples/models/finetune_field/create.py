import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import time
import numpy as np

import os
import pathlib

import torchfields

from finetune_loss import unsupervised_loss


def finetune(opti_loss, src, tgt, initial_res, lr, num_iter, opt_mode='adam',
            src_defects=None, tgt_defects=None,
            opt_params={}, crop=256, opt_res_coarsness=0, wd=1e-3, l2=0.0,
            gridsample_mode="bilinear"):

    pred_res = initial_res.clone()
    pred_res.requires_grad = True
    prev_pred_res = initial_res.clone()

    trainable = [pred_res]
    if opt_mode == 'adam':
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif opt_mode == 'sgd':
        optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)

    loss_bundle = {
        'src': src,
        'tgt': tgt,
    }
    if tgt_defects is not None:
        loss_bunlde['tgt_defects'] = tgt_defects
    if src_defects is not None:
        loss_bunlde['src_defects'] = src_defects

    prev_loss = []

    s = time.time()
    loss_bundle['pred_res'] = pred_res
    loss_bundle['pred_tgt'] = pred_res.from_pixels()(src)
    print (loss_bundle['pred_res'].abs().mean())

    loss_dict = opti_loss(loss_bundle, crop=crop)
    best_loss = loss_dict['result'].cpu().detach().numpy()

    new_best_ago = 0
    lr_halfed_count = 0
    nan_count = 0
    no_impr_count = 0
    new_best_count = 0
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

    for epoch in range(num_iter):
        loss_bundle['pred_tgt'] = pred_res.from_pixels()(src)
        loss_dict = opti_loss(loss_bundle, crop=crop)
        loss_var = loss_dict['result']
        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        loss_var += (pred_res**2).mean() * l2
        curr_loss = loss_var.cpu().detach().numpy()

        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        if np.isnan(curr_loss):
            nan_count += 1
            lr /= 1.5
            lr_halfed_count += 1
            pred_res = prev_pred_res.clone().detach()
            pred_res.requires_grad = True
            trainable = [pred_res]

            if opt_mode == 'adam':
                optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
            elif opt_mode == 'sgd':
                optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)

            prev_loss = []
            new_best_ago = 0
        else:
            min_improve = 1e-11
            if not np.isnan(curr_loss) and curr_loss + min_improve <= best_loss:
                prev_pred_res = pred_res.clone()
                best_loss = curr_loss
                #print ("new best")
                new_best_count += 1
                new_best_ago = 0
            else:
                new_best_ago += 1
                if new_best_ago > 12:
                    #print ("No improvement, reducing lr")
                    no_impr_count += 1
                    lr /= 2
                    lr_halfed_count += 1
                    if opt_mode == 'adam':
                        optimizer = torch.optim.Adam([pred_res], lr=lr)
                    elif opt_mode == 'sgd':
                        optimizer = torch.optim.SGD([pred_res], lr=lr, **opt_params)
                    new_best_ago -= 5
                prev_loss.append(curr_loss)

                optimizer.zero_grad()
                loss_var.backward()
                #torch.nn.utils.clip_grad_norm([pre_res, post_res], 4e0)
                pred_res.grad[pred_res.grad != pred_res.grad] = 0
                optimizer.step()
            if lr_halfed_count >= 15:
                break


    loss_bundle['pred_tgt'] = pred_res.from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)

    e = time.time()
    print ("New best: {}, No impr: {}, NaN: {}, Iter: {}".format(new_best_count, no_impr_count, nan_count, epoch))
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
    print (e - s)
    print ('==========')


    return pred_res



class Model(nn.Module):
    def __init__(self, lr, sm, downs_count, num_iter, crop, sm_keys_to_apply, mse_keys_to_apply):
        super().__init__()
        self.lr = lr
        self.downs_count = downs_count
        self.num_iter = num_iter
        self.crop = crop
        self.loss = unsupervised_loss(smoothness_factor=sm, use_defect_mask=True,
                                      sm_keys_to_apply=sm_keys_to_apply,
                                      mse_keys_to_apply=mse_keys_to_apply)


    def forward(self, src_img, tgt_img, pred_res, src_defects=None,
            tgt_defects=None, **kwargs):
        if pred_res.shape[1] != 2 and pred_res.shape[3] == 2:
            pred_res = pred_res.permute(0, 3, 1, 2)
        pred_res = pred_res.field()

        pred_res = finetune(self.loss, src_img, tgt_img, pred_res, self.lr, self.num_iter,
                )
        return pred_res



def create(**kwargs):
    return Model(**kwargs)
