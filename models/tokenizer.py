"""
Tokenizer (VAE) wrapper for encoding/decoding images.

This module provides:
1. VAE encoder (frozen) - for encoding real images to latents
2. VAE decoder (trainable) - for decoding latents to images

Following the same structure as VQ-Transplant's tokenizer design.
Uses Stable Diffusion VAE from HuggingFace.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from utils.hf_paths import resolve_sd_vae_source


# VAE normalization constants (same as iMF, SD VAE)
# Shape: (1, 4, 1, 1) for NCHW format
VAE_MEAN = torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409]).view(1, 4, 1, 1)
VAE_STD = torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003]).view(1, 4, 1, 1)


class Tokenizer(nn.Module):
    """
    VAE-based tokenizer for images.
    
    Encoder is frozen, Decoder is trainable for post-training.
    Uses Stable Diffusion VAE (sd-vae-ft-mse or sd-vae-ft-ema).
    """
    
    def __init__(self, args):
        """
        Initialize tokenizer from args.
        
        Args:
            args: Decoder configuration and VAE source.
        """
        super().__init__()
        self.args = args
        self.vae_type = getattr(args, 'vae_type', 'mse')
        
        print(f"[Tokenizer] Loading VAE (type={self.vae_type})...")
        vae_source, load_kwargs, load_message = resolve_sd_vae_source(self.vae_type, getattr(args, "vae_path", ""))
        print(load_message)
        self.vae = AutoencoderKL.from_pretrained(vae_source, **load_kwargs)
        if getattr(args, "decoder_init", "pretrained") == "random":
            fresh = AutoencoderKL.from_config(self.vae.config)
            self.vae.decoder.load_state_dict(fresh.decoder.state_dict(), strict=True)
            del fresh
            print("[Tokenizer] Randomly initialized decoder for initialization ablation")
        
        # Register normalization buffers
        self.register_buffer('mean', VAE_MEAN)
        self.register_buffer('std', VAE_STD)
        
        self.freeze_encoder_train_decoder()
        print("[Tokenizer] Loaded VAE: encoder frozen, decoder trainable")
    
    @torch.no_grad()
    def encode(self, x):
        """
        Encode image to normalized latent.
        
        Args:
            x: Image tensor, shape (B, 3, H, W) in range [-1, 1]
            
        Returns:
            Normalized latent, shape (B, 4, H/8, W/8)
        """
        latent_dist = self.vae.encode(x).latent_dist
        z = latent_dist.mean  # Use mean, not sample
        return (z - self.mean) / self.std
    
    def decode(self, z):
        """
        Decode normalized latent to image.
        
        Args:
            z: Normalized latent, shape (B, 4, H/8, W/8)
            
        Returns:
            Decoded image, shape (B, 3, H, W) in range [-1, 1]
        """
        z_denorm = z * self.std + self.mean
        return self.vae.decode(z_denorm).sample
    
    def get_decoder_params(self):
        """Get trainable decoder parameters."""
        decoder = self.vae.decoder
        # Handle DDP wrapper
        if hasattr(decoder, 'module'):
            decoder = decoder.module
        return list(decoder.parameters())

    def freeze_encoder_train_decoder(self):
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.vae.decoder.parameters():
            param.requires_grad = True
        self.vae.encoder.eval()
        self.vae.decoder.train()

    def encoder_eval(self):
        self.vae.encoder.eval()
    
    def train_decoder(self):
        """Set decoder to training mode."""
        self.vae.decoder.train()
        
    def eval_decoder(self):
        """Set decoder to evaluation mode."""
        self.vae.decoder.eval()
    
    def forward(self, x):
        """
        Full encode-decode forward pass.
        
        Args:
            x: Image tensor, shape (B, 3, H, W) in range [-1, 1]
            
        Returns:
            Reconstructed image, shape (B, 3, H, W)
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
    
    @property
    def last_layer(self):
        """Get last decoder layer for adaptive weight calculation."""
        decoder = self.vae.decoder
        if hasattr(decoder, 'module'):
            decoder = decoder.module
        return decoder.conv_out.weight


def build_tokenizer(args):
    return Tokenizer(args)
