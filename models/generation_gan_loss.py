import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminators import DinoDiscriminator
from utils.diff_aug import DiffAugment
from utils.util import Pack


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_gen_loss(logits_fake):
    return -torch.mean(logits_fake)


class LeCAMEMA:
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = (
            self.logits_real_ema * self.decay
            + torch.mean(logits_real).item() * (1.0 - self.decay)
        )
        self.logits_fake_ema = (
            self.logits_fake_ema * self.decay
            + torch.mean(logits_fake).item() * (1.0 - self.decay)
        )


def lecam_reg(logits_real, logits_fake, ema):
    real_reg = F.relu(logits_real - ema.logits_fake_ema).pow(2).mean()
    fake_reg = F.relu(ema.logits_real_ema - logits_fake).pow(2).mean()
    return real_reg + fake_reg


class GenerationGANLoss(nn.Module):
    """GAN-only objective for decoder adaptation on generated iMF latents."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lecam_ema = LeCAMEMA()
        self.discriminator = DinoDiscriminator()

    def forward(self, mode, real_images=None, fake_images=None):
        if mode == "generator":
            if fake_images is None:
                raise ValueError("fake_images is required for generator loss")
            return self.generator_loss(fake_images)
        if mode == "discriminator":
            if real_images is None or fake_images is None:
                raise ValueError("real_images and fake_images are required for discriminator loss")
            return self.discriminator_loss(real_images, fake_images)
        raise ValueError(f"Unknown GAN loss mode: {mode}")

    def generator_loss(self, fake_images):
        fake_aug = DiffAugment(
            fake_images.contiguous(),
            policy=self.args.diff_aug_policy,
            prob=self.args.diff_aug_prob,
        )
        logits_fake = self.discriminator(fake_aug)
        g_loss = hinge_gen_loss(logits_fake) * self.args.disc_weight
        return g_loss, Pack(
            gen_loss=g_loss,
            g_loss=g_loss.detach(),
            logits_fake=logits_fake.detach().mean(),
        )

    def discriminator_loss(self, real_images, fake_images):
        real_aug = DiffAugment(
            real_images.contiguous().detach(),
            policy=self.args.diff_aug_policy,
            prob=self.args.diff_aug_prob,
        )
        fake_aug = DiffAugment(
            fake_images.contiguous().detach(),
            policy=self.args.diff_aug_policy,
            prob=self.args.diff_aug_prob,
        )
        logits_real = self.discriminator(real_aug)
        logits_fake = self.discriminator(fake_aug)

        self.lecam_ema.update(logits_real, logits_fake)
        adv_loss = hinge_d_loss(logits_real, logits_fake)
        lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
        d_loss = self.args.disc_weight * (
            adv_loss + self.args.lecam_loss_weight * lecam_loss
        )

        cr_loss = torch.zeros((), device=real_images.device)
        if self.args.disc_cr_loss_weight > 0:
            real_aug_strong = DiffAugment(
                real_images.contiguous().detach(),
                policy=self.args.diff_aug_cr_policy,
                prob=1.0,
            )
            fake_aug_strong = DiffAugment(
                fake_images.contiguous().detach(),
                policy=self.args.diff_aug_cr_policy,
                prob=1.0,
            )
            logits_real_strong = self.discriminator(real_aug_strong)
            logits_fake_strong = self.discriminator(fake_aug_strong)
            cr_loss = F.mse_loss(
                torch.cat([logits_real, logits_fake], dim=0),
                torch.cat([logits_real_strong, logits_fake_strong], dim=0),
            )
            d_loss = d_loss + self.args.disc_cr_loss_weight * cr_loss

        return d_loss, Pack(
            d_loss=d_loss,
            d_adv_loss=adv_loss.detach(),
            d_lecam_loss=lecam_loss.detach(),
            d_cr_loss=cr_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
        )
