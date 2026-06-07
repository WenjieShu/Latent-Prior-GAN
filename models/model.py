"""Latent Prior GAN model."""

import torch
import torch.nn as nn

from models.diffusion_pytorch import DiffusionModelPyTorch
from models.losses import DinoGANLoss
from models.tokenizer import Tokenizer


class LatentPriorGAN(nn.Module):
    """Frozen iMF prior, trainable SD-VAE decoder, and DINO discriminator."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = Tokenizer(args)
        self.diffusion = DiffusionModelPyTorch(args)
        self.gan_loss = DinoGANLoss(args)
        self.freeze_prior_and_encoder()

    def load_prior(self):
        self.diffusion.load()
        self.diffusion.eval()

    def freeze_prior_and_encoder(self):
        for param in self.tokenizer.vae.parameters():
            param.requires_grad = False
        for param in self.tokenizer.vae.decoder.parameters():
            param.requires_grad = True
        for param in self.diffusion.parameters():
            param.requires_grad = False
        self.tokenizer.vae.encoder.eval()
        self.tokenizer.vae.decoder.train()
        self.diffusion.eval()

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.vae.encoder.eval()
        self.diffusion.eval()
        return self

    def decoder_parameters(self):
        return list(self.tokenizer.vae.decoder.parameters())

    def discriminator_parameters(self):
        return list(self.gan_loss.discriminator.parameters())

    def set_discriminator_requires_grad(self, requires_grad):
        for param in self.gan_loss.discriminator.parameters():
            param.requires_grad = requires_grad

    @torch.no_grad()
    def sample_latents(self, labels):
        return self.diffusion.generate(
            batch_size=labels.size(0),
            labels=labels,
            num_steps=self.args.num_steps,
            use_cfg=self.args.use_cfg,
            device=labels.device,
        )

    def decode(self, latents):
        return self.tokenizer.decode(latents).clamp(-1, 1)

    def forward(self, mode, real_images=None, labels=None):
        if labels is None:
            raise ValueError("labels are required")

        if mode == "generator":
            fake_images = self.decode(self.sample_latents(labels))
            return self.gan_loss.generator_loss(fake_images)

        if mode == "discriminator":
            if real_images is None:
                raise ValueError("real_images are required")
            with torch.no_grad():
                fake_images = self.decode(self.sample_latents(labels))
            return self.gan_loss.discriminator_loss(real_images, fake_images)

        raise ValueError(f"Unknown LatentPriorGAN mode: {mode}")
