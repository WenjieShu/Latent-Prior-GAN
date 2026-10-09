"""Latent Prior GAN model."""

import torch
import torch.nn as nn

from models.priors import build_prior
from models.tokenizer import build_tokenizer
from models.registry import validate_prior_inputs

class LatentPriorGenerator(nn.Module):
    """Frozen continuous latent prior, trainable SD-VAE decoder."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        validate_prior_inputs(args)
        self.tokenizer = build_tokenizer(args)
        self.diffusion = build_prior(args)
        if hasattr(self.diffusion, "validate_tokenizer"):
            self.diffusion.validate_tokenizer(self.tokenizer)
        self.freeze_prior_and_encoder()
        self.load_prior()

    def load_prior(self):
        if hasattr(self.diffusion, "load"):
            self.diffusion.load()
        self.diffusion.eval()

    def freeze_prior_and_encoder(self):
        self.tokenizer.freeze_encoder_train_decoder()
        for param in self.diffusion.parameters():
            param.requires_grad = False
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
    def paired_latents(self, real_images, labels):
        """Generation-aware paired supervision through a frozen continuous prior."""
        z_real = self.tokenizer.encode(real_images)
        z_denoised = self.diffusion.denoise_from_encoder_latent(
            z_real,
            maximum_noise_level=self.args.supervision_noise_level,
            normalized=self.args.supervision_normalized,
            labels=labels,
            num_steps=self.args.supervision_num_steps,
            use_cfg=self.args.supervision_use_cfg,
            fixed_noise=True,
            minimum_noise_level=self.args.supervision_noise_level,
        )
        return z_denoised

    def decode(self, latents):
        return self.tokenizer.decode(latents).clamp(-1, 1)


    ## pure gan mode / supervised gan mode
    def forward(self, x=None, labels=None):
        assert labels != None
        if x is None:
            x_gen  = self.decode(self.sample_latents(labels))
            return x_gen
        return self.generation_and_supervision(x, labels)

    ## supervised gan mode
    def generation_and_supervision(self, x, labels=None):
        assert labels != None
        x_gen  = self.decode(self.sample_latents(labels))
        z_denoised = self.paired_latents(x, labels)
        x_sup = self.decode(z_denoised)
        return x_gen, x_sup

