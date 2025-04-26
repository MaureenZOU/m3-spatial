#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from .point_features import coords_sample, point_sample
from xy_utils.memory import points_index_to_raw


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt, valid_mask=None, fdim=0):
    if valid_mask is None:
        # valid mask like gt shape but fdim is 1
        gt_shape = list(gt.shape)
        gt_shape[fdim] = 1
        valid_mask = torch.ones(gt_shape, device=gt.device)
        
    nd = gt.shape[fdim]
    loss_value = (((network_output - gt) ** 2 * valid_mask) * valid_mask).sum() / (valid_mask.sum() * nd)
    return loss_value


def cosine_loss(network_output, gt, valid_mask=None, dim=-1):
    if valid_mask is None:
        valid_mask = torch.ones_like(gt)
    valid_mask = valid_mask.mean(dim=dim)
    loss_value = (((1 - F.cosine_similarity(network_output, gt, dim=dim)) * valid_mask)).sum() / valid_mask.sum()
    return loss_value


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def point_sample_cos_with_mask(preds, gts, projs, mems, args, sample_number=2000):
    # fea1: pred, (C, H, W), fea2: gt, (H, W, C)
    losses = {}
    models = ['clip', 'llama3', 'siglip', 'dinov2', 'seem', 'llamav']
    for model in models:
        if getattr(args, f'use_{model}'):
            gt = gts[model]
            pred = preds[getattr(args, f'{model}_bit')[0]:getattr(args, f'{model}_bit')[1],:,:]
            sampled_points = coords_sample(pred.permute(1,2,0), sample_number)
            sampled_pred = point_sample(pred[None,], sampled_points[None,])[0]
            sampled_gt = point_sample(gt.permute(2,0,1)[None,], sampled_points[None,].type_as(gt))[0]
            raw_sampled_pred = points_index_to_raw(sampled_pred, projs[model], mems[model], _temp=args.softmax_temp).t()
            valide_mask = (sampled_gt.sum(dim=0, keepdim=True) != 0) * 1.0
            cos_dist = cosine_loss(raw_sampled_pred, sampled_gt, valide_mask, dim=0)
            l2_dist = l2_loss(raw_sampled_pred, sampled_gt, valide_mask)

            losses.update({
                f"{model}_cosine_loss": cos_dist,
                f"{model}_l2_loss": l2_dist
            })
    return losses

def pixelwise_l1_with_mask(img1, img2, pixel_mask):
    # img1, img2: (3, H, W)
    # pixel_mask: (H, W) bool torch tensor as mask.
    # only compute l1 loss for the pixels that are touched

    pixelwise_l1_loss = torch.abs((img1 - img2)) * pixel_mask.unsqueeze(0)
    return pixelwise_l1_loss


def pixelwise_ssim_with_mask(img1, img2, pixel_mask):
    window_size = 11

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0)

    return pixelwise_ssim_loss