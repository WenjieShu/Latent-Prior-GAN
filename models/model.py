"""Latent Prior GAN model."""

import torch
import torch.nn as nn

from models.diffusion_pytorch import DiffusionModelPyTorch
from models.tokenizer import Tokenizer


class LatentPriorGAN(nn.Module):
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
        if self.args.train_generator == False:
            for param in self.diffusion.parameters():
                param.requires_grad = False
        else:
            for param in self.diffusion.parameters():
                param.requires_grad = True

        self.tokenizer.vae.encoder.eval()
        self.tokenizer.vae.decoder.train()
        if self.args.train_generator == False:
            self.diffusion.eval()
        else:
            self.diffusion.train()

    def decoder_parameters(self):
        return list(self.tokenizer.vae.decoder.parameters())

    ### please refer to FD-loss paper, how to pass GAN loss gradient to generator 
    ### current version: generator cannot be updated, we must find one version that generator can be updated  
    @torch.no_grad()
    def sample_latents(self, labels):
        return self.diffusion.generate(
            batch_size=labels.size(0),
            labels=labels,
            num_steps=self.args.num_steps,
            use_cfg=self.args.use_cfg,
            device=labels.device,s
        )

    def decode(self, latents):
        return self.tokenizer.decode(latents).clamp(-1, 1)

    def forward(self, labels=None):
        assert labels != None
        x_gen  = self.decode(self.sample_latents(labels))
        return x_gen