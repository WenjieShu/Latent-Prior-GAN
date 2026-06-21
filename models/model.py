"""Latent Prior GAN model."""

import torch
import torch.nn as nn

from models.diffusion_pytorch import DiffusionModelPyTorch
from models.tokenizer import Tokenizer

class LatentPriorGenerator(nn.Module):
    """Frozen iMF prior, trainable SD-VAE decoder."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = Tokenizer(args)
        self.diffusion = DiffusionModelPyTorch(args)
        self.freeze_prior_and_encoder()
        self.load_prior()

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

     @torch.no_grad()
    def sample_latents(self, labels):
        return self.diffusion.generate(
            batch_size=labels.size(0),
            labels=labels,
            num_steps=self.args.num_steps,
            use_cfg=self.args.use_cfg,
            device=labels.device,
        )
    
    @torch.no_grad()
    def supervision_from_GAR(self, real_images, labels):
    """GAR-style reconstruction: E(x) -> noisy latent -> frozen iMF -> decoder."""
        z_real = self.tokenizer.encode(real_images)
        z_denoised = self.diffusion.denoise_from_encoder_latent(
            z_real,
            maximum_noise_level=self.args.supervision_noise_level,
            normalized=self.args.supervision_normalized,
            labels=labels,
            num_steps=self.args.supervision_num_steps,
            use_cfg=self.args.supervision_use_cfg,
            fixed_noise=True,
            minimum_noise_level=args.supervision_noise_level,
        )
    return z_denoised

    def decode(self, latents):
        return self.tokenizer.decode(latents).clamp(-1, 1)


    ## pure gan mode
    def forward(self, labels=None):
        assert labels != None
        x_gen  = self.decode(self.sample_latents(labels)) 
        return x_gen

    ## supervised gan mode
    def generation_and_supervison(self, x, labels=None):
        assert labels != None
        x_gen  = self.decode(self.sample_latents(labels)) 
        z_denoised = self.supervision_from_GAR(x, labels)
        x_sup = self.decode(z_denoised)
        return x_gen, x_sup
        