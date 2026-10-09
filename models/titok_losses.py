import torch
import torch.nn.functional as F
from torch import nn

from models.discriminators import DinoDiscriminator
from utils.diff_aug import DiffAugment
from utils.util import Pack


def hinge_d_loss(real_logits, fake_logits):
    return 0.5 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())


class ARPairwiseGANLoss(nn.Module):
    def __init__(self, pairwise_weight=1.0, disc_weight=1.0,
                 lecam_loss_weight=0.001, disc_cr_loss_weight=0.1):
        super().__init__()
        self.pairwise_weight = pairwise_weight
        self.disc_weight = disc_weight
        self.lecam_loss_weight = lecam_loss_weight
        self.disc_cr_loss_weight = disc_cr_loss_weight
        self.disc_adaptive_weight = True
        self.logits_real_ema = 0.0
        self.logits_fake_ema = 0.0
        self.lecam_decay = 0.999
        self.discriminator = DinoDiscriminator()

    def calculate_adaptive_weight(self, pairwise_loss, gan_loss, last_layer):
        pairwise_grads = torch.autograd.grad(
            pairwise_loss, last_layer, retain_graph=True, allow_unused=False
        )[0]
        gan_grads = torch.autograd.grad(
            gan_loss, last_layer, retain_graph=True, allow_unused=False
        )[0]
        weight = torch.norm(pairwise_grads) / (torch.norm(gan_grads) + 1e-4)
        return torch.clamp(weight, 0.0, 1e4).detach()

    def forward(self, real_images, generated_images, target_images,
                optimizer_idx, last_layer=None):
        fake_aug = DiffAugment(generated_images, policy="color,translation,cutout_0.2", prob=0.5)
        if optimizer_idx == 0:
            pairwise = F.mse_loss(real_images, target_images.detach())
            fake_logits = self.discriminator(fake_aug)
            gan = -fake_logits.mean()
            if self.disc_adaptive_weight:
                if last_layer is None:
                    raise ValueError("last_layer is required for adaptive adversarial weighting")
                adaptive_weight = self.calculate_adaptive_weight(
                    self.pairwise_weight * pairwise, gan, last_layer
                )
            else:
                adaptive_weight = pairwise.new_tensor(1.0)
            total = self.pairwise_weight * pairwise + adaptive_weight * self.disc_weight * gan
            return total, Pack(pairwise_loss=pairwise, g_loss=gan,
                               disc_adaptive_weight=adaptive_weight)
        real_aug = DiffAugment(target_images.detach(), policy="color,translation,cutout_0.2", prob=0.5)
        real_logits = self.discriminator(real_aug)
        fake_logits = self.discriminator(fake_aug.detach())
        real_mean = real_logits.mean().item()
        fake_mean = fake_logits.mean().item()
        self.logits_real_ema = self.lecam_decay * self.logits_real_ema + (1 - self.lecam_decay) * real_mean
        self.logits_fake_ema = self.lecam_decay * self.logits_fake_ema + (1 - self.lecam_decay) * fake_mean
        lecam = (
            F.relu(real_logits - self.logits_fake_ema).pow(2).mean()
            + F.relu(self.logits_real_ema - fake_logits).pow(2).mean()
        )
        hinge = hinge_d_loss(real_logits, fake_logits)
        d_loss = self.disc_weight * (hinge + self.lecam_loss_weight * lecam)
        real_strong = self.discriminator(
            DiffAugment(target_images.detach(), policy="color,translation,cutout_0.5", prob=1.0)
        )
        fake_strong = self.discriminator(
            DiffAugment(generated_images.detach(), policy="color,translation,cutout_0.5", prob=1.0)
        )
        consistency = F.mse_loss(
            torch.cat([real_logits, fake_logits], dim=0),
            torch.cat([real_strong, fake_strong], dim=0),
        )
        d_loss = d_loss + self.disc_cr_loss_weight * consistency
        return d_loss, Pack(d_loss=d_loss, logits_real=real_logits.mean().detach(),
                            logits_fake=fake_logits.mean().detach(),
                            hinge_d_loss=hinge, lecam_loss=lecam,
                            disc_cr_loss=consistency)
