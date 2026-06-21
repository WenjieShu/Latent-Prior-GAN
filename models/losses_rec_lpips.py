"""Reconstruction and LPIPS losses for Latent Prior GAN regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.perceptual_loss import PerceptualLoss
from utils.util import Pack


class RecLPIPSLoss(nn.Module):
    """GAR-style reconstruction anchor without discriminator terms."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rec_weight = float(args.rec_weight)
        self.lpips_weight = float(args.lpips_weight)
        self.perceptual_loss = PerceptualLoss(
            model_name=args.rec_perceptual_model
        ).eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs, reconstructions):
        rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous())
        lpips_loss = self.perceptual_loss(
            inputs.contiguous(), reconstructions.contiguous()
        )
        lpips_loss = torch.mean(lpips_loss)

        rec_lpips_loss = self.rec_weight * rec_loss + self.lpips_weight * lpips_loss
        loss_pack = Pack(
            rec_lpips_loss=rec_lpips_loss,
            rec_loss=rec_loss,
            lpips_loss=lpips_loss,
        )
        return rec_lpips_loss, loss_pack
