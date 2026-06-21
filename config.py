"""Configuration for Latent Prior GAN training."""

import argparse
import json
import os
import random

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - only needed when --config is used.
    yaml = None


MODEL_CFG = {
    "iMF-B-2": {"omega": 8.0, "t_min": 0.4, "t_max": 0.65},
    "iMF-M-2": {"omega": 10.5, "t_min": 0.4, "t_max": 0.6},
    "iMF-L-2": {"omega": 10.5, "t_min": 0.4, "t_max": 0.6},
    "iMF-XL-2": {"omega": 8.0, "t_min": 0.42, "t_max": 0.62},
}


def parse_arg():
    parser = argparse.ArgumentParser(description="Latent Prior GAN on iMF generative model.")
    parser.add_argument("--config", default="", type=str, help="optional YAML config")
    parser.add_argument("--dataset_dir", default="/mnt/nfs/wenjie/dataset/imagenet_idx")
    parser.add_argument("--dataset_name", default="ImageNet", choices=["ImageNet"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--model_type", default="iMF-XL-2", choices=["iMF-B-2", "iMF-M-2", "iMF-L-2", "iMF-XL-2"])
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--vae_type", default="mse", choices=["mse", "ema"])
    parser.add_argument("--pretrained_imf_pytorch", default="")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--cfg_omega", type=float, default=None)
    parser.add_argument("--cfg_t_min", type=float, default=None)
    parser.add_argument("--cfg_t_max", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--debug_steps", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr_decay_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume", default="", type=str)

    parser.add_argument("--disc_weight", type=float, default=0.5)
    parser.add_argument("--lecam_loss_weight", type=float, default=0.001)
    parser.add_argument("--disc_cr_loss_weight", type=float, default=4.0)
    parser.add_argument("--gan_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--rec_perceptual_model", default="lpips-convnext_s-1.0-0.1")
    parser.add_argument("--supervision_noise_level", type=float, default=0.1)
    parser.add_argument("--supervision_num_steps", type=int, default=1)
    parser.add_argument("--supervision_use_cfg", action="store_true", default=False)
    parser.add_argument("--supervision_normalized", action="store_true", default=False)

    parser.add_argument("--checkpoint_dir", default="/mnt/nfs/wenjie/Latent-Prior-GAN/checkpoints")
    parser.add_argument("--results_dir", default="/mnt/nfs/wenjie/Latent-Prior-GAN/results")
    parser.add_argument("--yaml_dir", default="/mnt/nfs/wenjie/Latent-Prior-GAN/yaml")
    parser.add_argument("--saver_name_pre", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="Latent-Prior-GAN")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--local-rank", default=-1, type=int)

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")

    config_args = _load_config(args.config)
    if config_args:
        defaults = {key: value for key, value in config_args.items() if hasattr(args, key)}
        parser.set_defaults(**defaults)
        args = parser.parse_args()

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.workers = min(max(0, args.workers), args.batch_size)
    _fill_defaults(args)
    _create_directories(args)
    _save_config(args)
    _set_seed(args.seed)
    return args


def _load_config(path):
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required when --config is used.")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _fill_defaults(args):
    if not args.pretrained_imf_pytorch:
        args.pretrained_imf_pytorch = (
            f"/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/{args.model_type}.pt"
        )
    if not args.saver_name_pre:
        cfg_tag = "cfg" if args.use_cfg else "nocfg"
        args.saver_name_pre = f"latent_prior_gan_{args.model_type}_nt1_gan_only_{cfg_tag}"

    cfg = MODEL_CFG[args.model_type]
    if args.cfg_omega is None:
        args.cfg_omega = cfg["omega"]
    if args.cfg_t_min is None:
        args.cfg_t_min = cfg["t_min"]
    if args.cfg_t_max is None:
        args.cfg_t_max = cfg["t_max"]


def _create_directories(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.yaml_dir, exist_ok=True)


def _save_config(args):
    path = os.path.join(args.yaml_dir, f"{args.saver_name_pre}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    args.config_path = path

def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    parsed_args = parse_arg()
    print("Configuration:")
    for key, value in sorted(vars(parsed_args).items()):
        print(f"  {key}: {value}")
