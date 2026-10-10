"""Latent Prior GAN model."""

import torch
import torch.nn as nn

from models.priors import build_prior
from models.registry import validate_prior_inputs
from models.tokenizer import build_tokenizer


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
        self.diffusion.requires_grad_(False)
        if hasattr(self.diffusion, "load"):
            self.diffusion.load()
        self.train_mode()

    def train_mode(self):
        """Train only the decoder, keeping the encoder and latent prior frozen."""
        self.train()
        self.tokenizer.train_mode()
        self.diffusion.eval()

    def eval_mode(self):
        self.eval()
        self.requires_grad_(False)

    def trainable_parameters(self):
        return self.tokenizer.trainable_parameters()

    @property
    def last_layer(self):
        return self.tokenizer.last_layer

    @torch.no_grad()
    def sample_latents(self, labels, seeds=None):
        return self.diffusion.generate(
            batch_size=labels.size(0),
            labels=labels,
            num_steps=self.args.num_steps,
            use_cfg=self.args.use_cfg,
            device=labels.device,
            seeds=seeds,
        )

    @torch.no_grad()
    def paired_latents(self, real_images, labels):
        """Generation-aware paired supervision through a frozen continuous prior."""
        z_real = self.tokenizer.encode(real_images)
        z_denoised = self.diffusion.denoise_from_encoder_latent(
            z_real,
            maximum_noise_level=self.args.supervision_noise_level,
            normalized=getattr(self.args, "supervision_normalized", False),
            labels=labels,
            num_steps=self.args.supervision_num_steps,
            use_cfg=self.args.supervision_use_cfg,
            fixed_noise=True,
            minimum_noise_level=self.args.supervision_noise_level,
        )
        return z_denoised

    def decode(self, latents):
        return self.tokenizer.decode(latents).clamp(-1, 1)

    def standard_generation(self, labels, seeds=None):
        return self.decode(self.sample_latents(labels, seeds))

    def sampling_metadata(self):
        if self.args.prior_family == "imf":
            return {
                "omega": self.args.cfg_omega if self.args.use_cfg else 1.0,
                "t_min": self.args.cfg_t_min if self.args.use_cfg else 0.0,
                "t_max": self.args.cfg_t_max if self.args.use_cfg else 1.0,
            }
        if self.args.prior_family == "dmf":
            return dict(self.diffusion.defaults)
        from omegaconf import OmegaConf

        return OmegaConf.to_container(self.diffusion.sampling, resolve=True)

    def forward(self, real_images=None, labels=None):
        """Return (generated_images, paired_images); w/o PS supplies no real images."""
        if labels is None:
            raise ValueError("Class labels are required for LP-GANs generation")
        generated_images = self.standard_generation(labels)
        paired_images = None
        if real_images is not None:
            paired_images = self.decode(self.paired_latents(real_images, labels))
        return generated_images, paired_images
