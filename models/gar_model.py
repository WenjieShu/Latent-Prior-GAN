"""
Combined model for decoder post-training.

This module combines:
1. DiffusionModel (frozen iMF) - for denoising latents
2. Tokenizer (VAE with trainable decoder) - for encoding/decoding

Following the same structure as VQ-Transplant's var_model.py

Pipeline:
    x (GT) → Encoder (frozen) → z_real 
                                  ↓
                            add_noise(t)
                                  ↓
                               z_noisy
                                  ↓
                        iMF Model (frozen)
                                  ↓
                              z_denoised
                                  ↓
                        Decoder (trainable) → x_rec
                                  ↓
                        Loss(x_rec, x) + GAN Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizer import Tokenizer


class TokenizerFlowComposition(nn.Module):
    """
    Main model for decoder post-training.
    
    Combines:
    - Tokenizer (VAE): encoder (frozen) + decoder (trainable)
    - DiffusionModel (iMF): frozen, used for denoising
    
    Uses the PyTorch-converted iMF checkpoint for denoising.
    """
    
    def __init__(self, args):
        """
        Initialize combined model.
        
        Args:
            args: Configuration with model_type, pretrained_imf, etc.
        """
        super().__init__()
        self.args = args
        
        # Initialize components
        print("[TokenizerFlowComposition] Initializing components...")

        if getattr(args, "debug_mode", False):
            self.diffusion = None
            self.diffusion_pytorch = None
        else:
            from models.diffusion_pytorch import DiffusionModelPyTorch
            self.diffusion = None
            self.diffusion_pytorch = DiffusionModelPyTorch(args)
            self.diffusion_pytorch.load()
            self.diffusion_pytorch.eval()
        
        # Tokenizer (PyTorch, decoder trainable)
        self.tokenizer = Tokenizer(args)
        
        print("[TokenizerFlowComposition] Initialized")
    
    def parameters(self):
        """Return all parameters (for counting)."""
        return self.tokenizer.parameters()
    
    def train_decoder(self):
        """Set decoder to training mode, encoder frozen."""
        self.tokenizer.train_decoder()
    
    def eval_mode(self):
        """Set entire model to evaluation mode."""
        self.tokenizer.eval()
        if self.diffusion_pytorch is not None:
            self.diffusion_pytorch.eval()
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.tokenizer = self.tokenizer.to(device)
        if self.diffusion_pytorch is not None:
            self.diffusion_pytorch = self.diffusion_pytorch.to(device)
        return self

    def decode_latent(self, z):
        return self.tokenizer.decode(z).clamp(-1, 1)
    
    # ============================================================
    # Pipeline 1: Vanilla VAE Reconstruction (for rFID baseline)
    # x → Encoder → z_real → Decoder → x_rec
    # ============================================================
    def vae_reconstruction(self, x):
        """
        Vanilla VAE reconstruction (no iMF).
        Used for computing baseline rFID.
        """
        with torch.no_grad():
            z_real = self.tokenizer.encode(x)
        x_rec = self.decode_latent(z_real)
        return x_rec
    
    # ============================================================
    # Pipeline 2: iMF Generation without CFG (for gFID baseline)
    # noise → iMF → z_gen → Decoder → x_gen
    # ============================================================
    def imf_image_generation_without_cfg(self, batch_size, labels=None, rng=None):
        """
        Generate images using iMF without CFG.
        Used for computing gFID without CFG.
        """
        z_gen = self.diffusion_pytorch.generate(
            batch_size=batch_size,
            labels=labels,
            num_steps=getattr(self.args, "num_steps", 1),
            use_cfg=False,
            device=next(self.tokenizer.parameters()).device,
        )
        x_gen = self.tokenizer.decode(z_gen)
        return x_gen.clamp(-1, 1)
    
    # ============================================================
    # Pipeline 3: iMF Generation with CFG (for gFID with CFG)
    # noise → iMF (CFG) → z_gen → Decoder → x_gen
    # ============================================================
    def imf_image_generation_with_cfg(self, batch_size, labels=None, rng=None):
        """
        Generate images using iMF with CFG.
        Used for computing gFID with CFG.
        """
        z_gen = self.diffusion_pytorch.generate(
            batch_size=batch_size,
            labels=labels,
            num_steps=getattr(self.args, "num_steps", 1),
            use_cfg=True,
            device=next(self.tokenizer.parameters()).device,
        )
        x_gen = self.tokenizer.decode(z_gen)
        return x_gen.clamp(-1, 1)
    
    # ============================================================
    # Pipeline 4: Our Proposed Reconstruction (for training)
    # x → Encoder → z_real → add_noise(t) → z_noisy → iMF → z_denoised → Decoder → x_rec
    # ============================================================
    def vae_imf_reconstruction(self, x, rng=None, labels=None, use_cfg=False):
        """
        Our proposed reconstruction pipeline for training.
        
        Args:
            x: Input images, shape (B, 3, H, W) in range [-1, 1]
            rng: JAX random key for noise (only used for JAX backend)
            labels: Class labels for conditional generation
            use_cfg: Whether to use CFG for denoising (False for reconstruction)
            
        Returns:
            x_rec: Reconstructed images
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode (frozen)
        with torch.no_grad():
            z_real = self.tokenizer.encode(x)  # (B, 4, H/8, W/8) NCHW
        
        if self.diffusion_pytorch is not None:
            with torch.no_grad():
                z_denoised = self.diffusion_pytorch.denoise_from_encoder_latent(
                    z_real,
                    maximum_noise_level=self.args.maximum_noise_level,
                    normalized=getattr(self.args, 'normalized', False),
                    labels=labels,
                    num_steps=1,
                    use_cfg=use_cfg,
                    fixed_noise=getattr(self.args, 'fixed_noise', False)
                )
        else:
            z_denoised = z_real
        
        # Decode (trainable)
        x_rec = self.tokenizer.decode(z_denoised)
        return x_rec.clamp(-1, 1)
    
    def forward(self, x, rng=None, labels=None):
        """
        Forward pass for training.
        
        Args:
            x: Input images, shape (B, 3, H, W) in range [-1, 1]
            rng: JAX random key (only used for JAX backend)
            labels: Class labels (must be provided from dataloader)
            
        Returns:
            x_rec: Reconstructed images from our pipeline
        """
        batch_size = x.size(0)
        device = x.device
        
        assert labels is not None, (
            "labels must be provided from the dataloader, "
            "do not use random labels for conditional denoising"
        )
        
        return self.vae_imf_reconstruction(x, rng=None, labels=labels)
    
    def collect_eval_info(self, x, rng=None, labels=None):
        """
        Collect evaluation metrics.
        
        Returns:
            x_rec: Reconstructed images
            rec_loss: Reconstruction MSE loss
        """
        x_rec = self.vae_imf_reconstruction(x, rng=rng, labels=labels)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
    
    @property
    def last_layer(self):
        """Get last decoder layer for adaptive weight calculation."""
        return self.tokenizer.last_layer
