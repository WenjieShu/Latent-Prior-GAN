"""Frozen DMF-XL/2 adapter with no changes to upstream DMF source."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .latent_bridge import dmf_to_lpg, lpg_to_dmf
from .registry import prior_defaults


def _import_upstream(dmf_root):
    root = Path(dmf_root).resolve()
    if not (root / "models" / "dmft.py").is_file():
        raise FileNotFoundError(f"Invalid DMF repository: {root}")
    package = "lpg_external_dmf"
    # DMF's optional JVP attention kernels retain one absolute import
    # (``models.triton_utils``).  Extend the already-loaded LPG ``models``
    # package with DMF's module directory so that import resolves to the
    # upstream helper without replacing the LPG package or its training code.
    import models as lpg_models
    dmf_models_path = str(root / "models")
    if dmf_models_path not in lpg_models.__path__:
        lpg_models.__path__.append(dmf_models_path)
    if package not in sys.modules:
        top = types.ModuleType(package)
        top.__path__ = [str(root)]
        sys.modules[package] = top
        models = types.ModuleType(f"{package}.models")
        models.__path__ = [str(root / "models")]
        sys.modules[f"{package}.models"] = models
    dmft = importlib.import_module(f"{package}.models.dmft")
    samplers = importlib.import_module(f"{package}.samplers")
    return dmft, samplers


class DmfPrior(nn.Module):
    def __init__(self, args):
        super().__init__()
        defaults = prior_defaults("dmf")
        if args.vae_type != defaults["required_vae_type"]:
            raise ValueError("DMF requires --vae_type mse for decoder compatibility")
        if not args.prior_repo_path or not args.prior_checkpoint:
            raise ValueError("DMF requires --prior_repo_path and --prior_checkpoint")
        self.prior_dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }[args.prior_dtype]
        dmft, samplers = _import_upstream(args.prior_repo_path)
        self._euler_sampler = samplers.euler_sampler
        self.model = dmft.DMFT_models[defaults["model_name"]](
            input_size=args.latent_size,
            num_classes=1000,
            use_cfg_embedding=True,
            dmf_depth=defaults["dmf_depth"],
            use_logvar=True,
            attn_func="torch_sdpa",
            qk_norm=False,
        )
        state = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"DMF checkpoint mismatch: {incompatible}")
        self.defaults = defaults
        self.num_classes = 1000
        self.model.to(dtype=self.prior_dtype).eval().requires_grad_(False)

    def load(self):
        return self

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def _labels(self, batch_size, labels, device):
        return labels if labels is not None else torch.randint(0, self.num_classes, (batch_size,), device=device)

    @torch.no_grad()
    def generate(self, batch_size, labels=None, num_steps=1, use_cfg=True, device="cuda", seeds=None, **_):
        labels = self._labels(batch_size, labels, device)
        if seeds is None:
            z = torch.randn(
                batch_size, 4, 32, 32, device=device, dtype=self.prior_dtype,
            )
        else:
            z = torch.stack([
                torch.randn(4, 32, 32, generator=torch.Generator("cpu").manual_seed(int(seed)), dtype=torch.float32)
                for seed in seeds.cpu().tolist()
            ]).to(device=device, dtype=self.prior_dtype)
        cfg_scale = self.defaults["cfg_scale"] if use_cfg else 1.0
        out = self._euler_sampler(
            model=self.model, latents=z, y=labels,
            num_steps=num_steps, cfg_scale=cfg_scale,
            g_min=self.defaults["g_min"], g_max=self.defaults["g_max"],
            shift=self.defaults["shift"],
        )
        return dmf_to_lpg(out.float(), self.defaults["native_latent_scale"])

    @torch.no_grad()
    def denoise_from_encoder_latent(
        self, z_real, maximum_noise_level, labels=None, num_steps=1,
        use_cfg=False, fixed_noise=True, minimum_noise_level=0.0, normalized=False,
    ):
        if normalized:
            raise ValueError("DMF adapter does not support LPG per-sample normalized supervision")
        if not fixed_noise or float(minimum_noise_level) != float(maximum_noise_level):
            raise ValueError("DMF supervised adapter currently requires a fixed noise level")
        t = float(maximum_noise_level)
        z_clean = lpg_to_dmf(
            z_real, self.defaults["native_latent_scale"],
        ).to(dtype=self.prior_dtype)
        z_t = (1.0 - t) * z_clean + t * torch.randn_like(z_clean)
        labels = self._labels(len(z_t), labels, z_t.device)
        t_tensor = torch.full((len(z_t),), t, device=z_t.device, dtype=z_t.dtype)
        r_tensor = torch.zeros_like(t_tensor)
        velocity = self.model(z_t, t_tensor, r_tensor, labels)
        z_denoised = z_t - t * velocity
        return dmf_to_lpg(z_denoised.float(), self.defaults["native_latent_scale"])
