import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminators import DinoDiscriminator
from utils.diff_aug import DiffAugment
from utils.util import Pack


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


class LeCAM_EMA(object):
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(
            logits_real
        ).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(
            logits_fake
        ).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


class DinoGANLoss(nn.Module):
    """LP-GANs w/o PS: unweighted hinge G, weighted hinge/LeCAM/CR D."""

    def __init__(self, args):
        super(DinoGANLoss, self).__init__()
        self.args = args
        self.lecam_ema = LeCAM_EMA()
        self.discriminator = DinoDiscriminator()

    def forward(self, inputs, generations, optimizer_idx):
        # generator update
        if optimizer_idx == 0:
            generation_aug = DiffAugment(
                generations.contiguous(), policy="color,translation,cutout_0.2", prob=0.5
            )
            logits_fake = self.discriminator(generation_aug.contiguous())
            g_loss = hinge_gen_loss(logits_fake)
            gen_loss = g_loss
            loss_pack = Pack(gen_loss=gen_loss, g_loss=g_loss)
            return gen_loss, loss_pack

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(
                DiffAugment(
                    inputs.contiguous().detach(), policy="color,translation,cutout_0.2", prob=0.5
                )
            )
            logits_fake = self.discriminator(
                DiffAugment(
                    generations.contiguous().detach(),
                    policy="color,translation,cutout_0.2",
                    prob=0.5,
                )
            )

            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            adversarial_loss = hinge_d_loss(logits_real, logits_fake)
            d_loss = self.args.disc_weight * (
                lecam_loss * self.args.lecam_loss_weight + adversarial_loss
            )

            logits_real_s = self.discriminator(
                DiffAugment(
                    inputs.contiguous().detach(), policy="color,translation,cutout_0.5", prob=1.0
                )
            )
            logits_fake_s = self.discriminator(
                DiffAugment(
                    generations.contiguous().detach(),
                    policy="color,translation,cutout_0.5",
                    prob=1.0,
                )
            )
            disc_cr_loss_weight = self.args.disc_cr_loss_weight
            d_cr = (
                F.mse_loss(
                    torch.cat([logits_real, logits_fake], dim=0),
                    torch.cat([logits_real_s, logits_fake_s]),
                )
                * disc_cr_loss_weight
            )
            d_loss += d_cr

            logits_real = logits_real.detach().mean()
            logits_fake = logits_fake.detach().mean()
            loss_pack = Pack(
                d_loss=d_loss,
                logits_real=logits_real,
                logits_fake=logits_fake,
                hinge_d_loss=adversarial_loss,
            )

            return d_loss, loss_pack
