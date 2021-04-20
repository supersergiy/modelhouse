import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import six
import copy
from collections import defaultdict

from pdb import set_trace as st
from finetune_masks import get_mse_and_smoothness_masks


def field_dx(f, forward=False):
    if forward:
        delta = f[:,1:-1,:,:] - f[:,2:,:,:]
    else:
        delta = f[:,1:-1,:,:] - f[:,:-2,:,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 0, 0, 1, 1, 0, 0))
    return result

def field_dy(f, forward=False):
    if forward:
        delta = f[:,:,1:-1,:] - f[:,:,2:,:]
    else:
        delta = f[:,:,1:-1,:] - f[:,:,:-2,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 0, 0, 0, 0))
    return result

def field_dxy(f, forward=False):
    if forward:
        delta = f[:,1:-1,1:-1,:] - f[:,2:,2:,:]
    else:
        delta = f[:,1:-1,1:-1,:] - f[:,:-2,:-2,:]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result

def field_dxy2(f, forward=False):
    if forward:
        delta = f[:, 1:-1, 1:-1, :] - f[:, 2:, :-2, :]
    else:
        delta = f[:, 1:-1, 1:-1, :] - f[:, :-2, 2:, :]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result

def rigidity_score(field_delta, tgt_length, power=2):
    spring_lengths = torch.sqrt(field_delta[..., 0]**2 + field_delta[..., 1]**2 + 1e-8)
    spring_deformations = (spring_lengths - tgt_length).abs() ** power
    return spring_deformations

def pix_identity(size, batch=1, device='cuda'):
    result = torch.zeros((batch, size, size, 2), device=device)
    x = torch.arange(size, device=device)
    result[:, :, :, 1] = x
    result = torch.transpose(result, 1, 2)
    result[:, :, :, 0] = x
    result = torch.transpose(result, 1, 2)
    return result

def rigidity(field, power=2, diagonal_mult=0.8, two_diagonals=True):
    identity = pix_identity(size=field.shape[-2])
    field_abs = field + identity
    result = rigidity_score(field_dx(field_abs, forward=False), 1, power=power)
    result += rigidity_score(field_dx(field_abs, forward=True), 1, power=power)
    result += rigidity_score(field_dy(field_abs, forward=False), 1, power=power)
    result += rigidity_score(field_dy(field_abs, forward=True), 1, power=power)
    result += rigidity_score(field_dxy(field_abs, forward=True), 2**(1/2), power=power) * diagonal_mult
    result += rigidity_score(field_dxy(field_abs, forward=False), 2**(1/2), power=power) * diagonal_mult
    total = 4 + 2*diagonal_mult
    if two_diagonals:
        result += rigidity_score(field_dxy2(field_abs, forward=True), 2**(1/2), power=power) * diagonal_mult
        result += rigidity_score(field_dxy2(field_abs, forward=False), 2**(1/2), power=power) * diagonal_mult
        total += 2*diagonal_mult

    result /= total

    #compensate for padding
    result[..., 0:6, :] = 0
    result[..., -6:, :] = 0
    result[..., :,  0:6] = 0
    result[..., :, -6:] = 0

    return result.squeeze()

def smoothness_penalty(ptype='rig'):
    def penalty(field, weights=None):
        if ptype == 'rig':
            if field.shape[1] == 2 and field.shape[-1] != 2:
                field = field.permute(0, 2, 3, 1)

            field = rigidity(field)
        else:
            raise ValueError("Invalid penalty type: {}".format(ptype))

        return field
    return penalty


def similarity_score(bundle, weights=None, crop=32):
    tgt = bundle['tgt']
    pred_tgt = bundle['pred_tgt']

    mse = ((tgt - pred_tgt)**2)
    if crop > 0:
        mse = mse[..., crop:-crop, crop:-crop]
    if weights is not None:
        weights = weights
        if crop > 0:
            weights = weights[..., crop:-crop, crop:-crop]
        total_mse = torch.sum(mse * weights)
        mask_sum  = torch.sum(weights)
        if mask_sum == 0:
            return total_mse
        else:
            return total_mse / mask_sum
    else:
        return torch.mean(mse)

def smoothness_score(bundle, smoothness_type,
                     weights=None, crop=8):
    pixelwise = smoothness_penalty(smoothness_type)(bundle['pred_res'])
    if crop > 0:
        pixelwise = pixelwise[..., crop:-crop, crop:-crop]

    if weights is not None:
        weights = weights
        if crop > 0:
            weights = weights[..., crop:-crop, crop:-crop]
        total_sm = torch.sum(pixelwise * weights)
        mask_sum  = torch.sum(weights)
        if mask_sum == 0:
            return total_sm
        else:
            return total_sm / mask_sum
    else:
        return torch.mean(pixelwise)


def unsupervised_loss(smoothness_factor, smoothness_type='rig', use_defect_mask=False,
                      reverse=True, sm_keys_to_apply={}, mse_keys_to_apply={}):
    def compute_loss(bundle, smoothness_mult=1.0, crop=32):
        loss_dict = {}
        if use_defect_mask:
            mse_mask, smoothness_mask = get_mse_and_smoothness_masks(bundle,
                    sm_keys_to_apply=sm_keys_to_apply,
                    mse_keys_to_apply=mse_keys_to_apply)
        else:
            mse_mask = None
            smoothness_mask = None

        similarity = similarity_score(bundle,
                                      weights=mse_mask,
                                      crop=crop)
        if smoothness_mult != 0:
            smoothness = smoothness_score(bundle,
                                      weights=smoothness_mask,
                                      smoothness_type=smoothness_type,
                                      crop=crop)
        else:
            smoothness = torch.zeros(1, device=bundle['src'].device, dtype=torch.float32)
        result =  similarity + smoothness * smoothness_factor

        loss_dict['result'] = result
        loss_dict['similarity'] = similarity
        loss_dict['smoothness'] = smoothness * smoothness_factor * smoothness_mult
        loss_dict['vec_magnitude'] = torch.mean(torch.abs(bundle['pred_res']))
        loss_dict['vec_sim'] = torch.cuda.FloatTensor([0])
        if 'res' in bundle:
            loss_dict['vec_sim'] = torch.mean(torch.abs(bundle['pred_res'] - bundle['res']))
        loss_dict['mse_mask'] = mse_mask
        loss_dict['smoothness_mask'] = smoothness_mask
        return loss_dict
    return compute_loss


