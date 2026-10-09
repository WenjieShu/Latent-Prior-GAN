"""Frozen AlphaFlow-XL/2+ adapter around the official snapshot loader."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
from torch import nn

from .latent_bridge import lpg_to_raw, raw_to_lpg
from .registry import prior_defaults


class AlphaFlowPrior(nn.Module):
    def __init__(self, args):
        super().__init__()
        defaults = prior_defaults(args.prior_family)
        if args.vae_type != defaults["required_vae_type"]:
            raise ValueError("AlphaFlow requires --vae_type ema")
        for value, name in (
            (args.prior_repo_path, "--prior_repo_path"),
            (args.prior_checkpoint, "--prior_checkpoint"),
            (args.prior_vae_checkpoint, "--prior_vae_checkpoint"),
        ):
            if not value:
                raise ValueError(f"AlphaFlow requires {name}")
        root = Path(args.prior_repo_path).resolve()
        if not (root / "src" / "inference.py").is_file():
            raise FileNotFoundError(f"Invalid AlphaFlow repository: {root}")
        alpha_vae_path = getattr(args, "vae_path", "") or os.environ.get("ALPHAFLOW_SD_VAE_EMA_PATH")
        if alpha_vae_path:
            os.environ["ALPHAFLOW_SD_VAE_EMA_PATH"] = str(Path(alpha_vae_path).resolve())
        if not alpha_vae_path:
            alpha_vae_path = str(
                Path(args.prior_vae_checkpoint).resolve().parent.parent / "sd-vae-ft-ema"
            )
            os.environ["ALPHAFLOW_SD_VAE_EMA_PATH"] = alpha_vae_path
        if not Path(alpha_vae_path).is_dir():
            raise FileNotFoundError(
                "AlphaFlow needs the local diffusers EMA-VAE repository; set "
                f"ALPHAFLOW_SD_VAE_EMA_PATH (resolved value: {alpha_vae_path})"
            )
        sys.path.insert(0, str(root))
        from omegaconf import OmegaConf
        from src.training.network_utils import load_snapshot

        snapshot_cfg = OmegaConf.create({
            "snapshot_path": str(args.prior_checkpoint),
            "load_state": True,
            "use_ema": True,
            "allow_missing_keys": False,
            "allow_unexpected_keys": False,
            "convert_params_to_buffers": False,
            "fsdp": False,
            "cfg_overwrites": {
                "model": {"autoencoder_ckpt": {"snapshot_path": str(args.prior_vae_checkpoint)}},
                "training": {"dp": {"strategy": None}},
            },
        })
        net, _, experiment_cfg = load_snapshot(snapshot_cfg, verbose=True, device="cpu")
        self.net = net.eval().requires_grad_(False)
        self.sampling = copy.deepcopy(experiment_cfg.sampling)
        self.sampling.num_steps = defaults["num_steps"]
        self.sampling.sigma_noise = defaults["sigma_noise"]
        self.sampling.enable_trajectory_sampling = defaults["enable_trajectory_sampling"]
        self.sampling.enable_consistency_sampling = defaults["enable_consistency_sampling"]
        print(
            "AlphaFlow sampling CFG: "
            f"scale={float(self.sampling.cfg.scale)}, "
            f"t_min={float(self.sampling.cfg.t_min)}, "
            f"t_max={float(self.sampling.cfg.t_max)}"
        )
        self.num_classes = 1000

    def load(self):
        return self

    def validate_tokenizer(self, tokenizer):
        if not hasattr(self.net, "autoencoder"):
            raise RuntimeError("AlphaFlow snapshot was loaded without its validation VAE")
        public_vae = self.net.autoencoder.model.model
        public_state = public_vae.state_dict()
        tokenizer_state = tokenizer.vae.state_dict()
        if public_state.keys() != tokenizer_state.keys():
            missing = sorted(public_state.keys() - tokenizer_state.keys())
            extra = sorted(tokenizer_state.keys() - public_state.keys())
            raise RuntimeError(
                f"AlphaFlow/tokenizer EMA-VAE keys differ: missing={missing[:5]} extra={extra[:5]}"
            )
        max_abs = 0.0
        for key in public_state:
            lhs = public_state[key].detach().float().cpu()
            rhs = tokenizer_state[key].detach().float().cpu()
            max_abs = max(max_abs, (lhs - rhs).abs().max().item())
        if max_abs > 1e-6:
            raise RuntimeError(f"AlphaFlow EMA-VAE differs from tokenizer VAE: max_abs={max_abs}")
        print(f"AlphaFlow EMA-VAE identity check passed: max_abs={max_abs}")
        # Generation and the latent bridge only need the snapshot's latent
        # statistics after this identity check.  Keeping a second VAE would
        # waste several hundred MB on every DDP rank.
        del self.net.autoencoder

    def train(self, mode=True):
        super().train(False)
        self.net.eval()
        return self

    def _cond(self, labels):
        one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=self.num_classes).float()
        return self.net.construct_cond_from_labels(one_hot)

    def _velocity(self, z_t, sigma, sigma_next, cond, use_cfg):
        velocity = self.net(z_t, sigma=sigma, sigma_next=sigma_next, cond=cond)
        time_value = float(sigma.flatten()[0].item())
        inside_guidance_window = (
            float(self.sampling.cfg.t_min) <= time_value <= float(self.sampling.cfg.t_max)
        )
        scale = float(self.sampling.cfg.scale) if use_cfg and inside_guidance_window else 1.0
        if scale != 1.0:
            uncond = self.net(
                z_t, sigma=sigma, sigma_next=sigma_next,
                cond=cond, force_label_dropout=True,
            )
            velocity = uncond + scale * (velocity - uncond)
        return velocity

    def _alpha_to_lpg(self, z_alpha):
        raw = self.net.denormalize_latents(z_alpha.unsqueeze(1)).squeeze(1)
        return raw_to_lpg(raw)

    def _lpg_to_alpha(self, z_lpg):
        raw = lpg_to_raw(z_lpg).unsqueeze(1)
        return self.net.normalize_latents(raw).squeeze(1)

    @torch.no_grad()
    def generate(self, batch_size, labels=None, num_steps=1, use_cfg=True, device="cuda", seeds=None, **_):
        if num_steps != 1:
            raise ValueError("AlphaFlow adapter is locked to the released NFE=1 protocol")
        labels = labels if labels is not None else torch.randint(0, self.num_classes, (batch_size,), device=device)
        cond = self._cond(labels)
        if seeds is None:
            noise = torch.randn(batch_size, *self.net.input_shape, device=device)
        else:
            noise = torch.stack([
                torch.randn(*self.net.input_shape, generator=torch.Generator("cpu").manual_seed(int(seed)))
                for seed in seeds.cpu().tolist()
            ]).to(device)
        sigma = torch.ones(batch_size, 1, 1, 1, 1, device=device)
        sigma_next = torch.zeros_like(sigma)
        velocity = self._velocity(noise, sigma, sigma_next, cond, use_cfg)
        out = (noise - velocity).squeeze(1)
        return self._alpha_to_lpg(out)

    @torch.no_grad()
    def denoise_from_encoder_latent(
        self, z_real, maximum_noise_level, labels=None, num_steps=1,
        use_cfg=False, fixed_noise=True, minimum_noise_level=0.0, normalized=False,
    ):
        if normalized or not fixed_noise or float(minimum_noise_level) != float(maximum_noise_level):
            raise ValueError("AlphaFlow adapter requires fixed, non-normalized supervision")
        if num_steps != 1:
            raise ValueError("AlphaFlow supervised adapter currently supports one step")
        labels = labels if labels is not None else torch.randint(0, self.num_classes, (len(z_real),), device=z_real.device)
        cond = self._cond(labels)
        t = float(maximum_noise_level)
        clean = self._lpg_to_alpha(z_real).unsqueeze(1)
        z_t = (1.0 - t) * clean + t * self.sampling.sigma_noise * torch.randn_like(clean)
        sigma = torch.full((len(z_t), 1, 1, 1, 1), t, device=z_t.device)
        sigma_next = torch.zeros_like(sigma)
        velocity = self._velocity(z_t, sigma, sigma_next, cond, use_cfg)
        out = z_t - t * velocity
        return self._alpha_to_lpg(out.squeeze(1))
