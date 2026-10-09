"""Conversions between LPG-normalized and upstream native VAE latents."""

import torch


LPG_MEAN = torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409]).view(1, 4, 1, 1)
LPG_STD = torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003]).view(1, 4, 1, 1)


def lpg_to_raw(z_lpg):
    mean = LPG_MEAN.to(device=z_lpg.device, dtype=z_lpg.dtype)
    std = LPG_STD.to(device=z_lpg.device, dtype=z_lpg.dtype)
    return z_lpg * std + mean


def raw_to_lpg(z_raw):
    mean = LPG_MEAN.to(device=z_raw.device, dtype=z_raw.dtype)
    std = LPG_STD.to(device=z_raw.device, dtype=z_raw.dtype)
    return (z_raw - mean) / std


def lpg_to_dmf(z_lpg, scale=0.18215):
    return lpg_to_raw(z_lpg) * scale


def dmf_to_lpg(z_dmf, scale=0.18215):
    return raw_to_lpg(z_dmf / scale)
