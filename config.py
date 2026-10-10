"""LP-GANs training arguments; model defaults live in registries."""

import argparse
import json
import os
import random
from pathlib import Path

from models.registry import FAMILIES, IMF_MODELS, resolve_continuous_defaults


def load_yaml_config(path):
    """Read YAML defaults without constructing models or preparing a run."""
    if not path:
        return {}
    import yaml

    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("YAML config must be a mapping")
    return values


def resolve_family(values, cli_family=None):
    family = cli_family or values.get("prior_family", "imf")
    if family not in FAMILIES:
        raise ValueError(f"Unknown prior family: {family}")
    return family


def build_parser(family="imf"):
    titok = family == "titok"
    p = argparse.ArgumentParser(
        description="Train LP-GANs; --wo_ps selects LP-GANs w/o paired supervision (PS). Use --prior_family titok --help for token-pool options."
    )
    p.add_argument(
        "--config", default="", help="YAML defaults; explicit CLI values take precedence"
    )
    p.add_argument("--dataset_dir", default="data/imagenet")
    p.add_argument("--dataset_name", default="ImageNet", choices=["ImageNet"])
    p.add_argument("--batch_size", type=int, default=2 if titok else 8, help="per GPU")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256, choices=[256])
    p.add_argument("--prior_family", default=family, choices=FAMILIES)
    p.add_argument(
        "--wo_ps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LP-GANs w/o PS ablation (validated continuous-prior path)",
    )
    if titok:
        group = p.add_argument_group("TiTok token pools and decoder")
        group.add_argument("--real_latent_pool", default="")
        group.add_argument("--generated_latent_pool", default="")
        group.add_argument(
            "--tokenizer_name", default="titok-vql32", choices=["titok-vql32", "titok-vqb64"]
        )
        group.add_argument("--tokenizer_ckpt_path", default="")
        group.add_argument("--lr_decay", type=float, default=0.5)
        group.add_argument("--min_lr", type=float, default=1e-6)
        group.add_argument("--pairwise_weight", type=float, default=1.0)
    else:
        group = p.add_argument_group("Continuous latent prior and paired supervision")
        if family == "imf":
            group.add_argument("--model_type", default="iMF-B-2", choices=list(IMF_MODELS))
            group.add_argument("--pretrained_imf_pytorch", default="", help="official iMF .pth")
            group.add_argument("--cfg_omega", type=float, default=None)
            group.add_argument("--cfg_t_min", type=float, default=None)
            group.add_argument("--cfg_t_max", type=float, default=None)
            group.add_argument(
                "--supervision_normalized", action=argparse.BooleanOptionalAction, default=False
            )
        else:
            group.add_argument(
                "--prior_repo_path",
                default="",
                help="native DMF or AlphaFlow/MeanFlow source checkout",
            )
            group.add_argument("--prior_checkpoint", default="")
            if family == "dmf":
                group.add_argument("--latent_size", type=int, default=32, choices=[32])
                group.add_argument("--prior_dtype", default="fp32", choices=["fp32", "bf16"])
            else:
                group.add_argument(
                    "--prior_vae_checkpoint", default="", help="MeanFlow native EMA VAE snapshot"
                )
        group.add_argument("--vae_type", default=None, choices=["mse", "ema"])
        group.add_argument(
            "--vae_path",
            default="",
            help="local diffusers VAE directory; otherwise SD_VAE_PATH or HF",
        )
        group.add_argument("--decoder_init", default="pretrained", choices=["pretrained", "random"])
        group.add_argument("--num_steps", type=int, default=1)
        group.add_argument("--use_cfg", action=argparse.BooleanOptionalAction, default=True)
        group.add_argument("--supervision_noise_level", type=float, default=0.2)
        group.add_argument("--supervision_num_steps", type=int, default=1)
        group.add_argument(
            "--supervision_use_cfg", action=argparse.BooleanOptionalAction, default=True
        )
        group.add_argument(
            "--reset_resume_lr",
            action="store_true",
            help="after restoring optimizer states, apply --lr/--disc_lr",
        )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--debug_steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--disc_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0 if titok else 1e-4)
    p.add_argument("--disc_weight", type=float, default=1.0 if titok else 0.65)
    p.add_argument("--lecam_loss_weight", type=float, default=0.001)
    p.add_argument("--disc_cr_loss_weight", type=float, default=0.1 if titok else 4.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--checkpoint_dir", default="checkpoints/runs")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--yaml_dir", default="runs/configs")
    p.add_argument("--saver_name_pre", default="")
    p.add_argument("--seed", type=int, default=3407 if titok else 42)
    if not titok:
        p.add_argument("--use_wandb", action="store_true")
        p.add_argument("--wandb_project", default="Latent-Prior-GAN")
        p.add_argument("--wandb_run_name", default="")
        p.add_argument("--dist-backend", default="nccl")
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    return p


def apply_yaml_defaults(parser, values):
    """Validate YAML keys, types, and choices before applying CLI overrides."""
    actions = {action.dest: action for action in parser._actions}
    unknown = set(values) - (set(actions) - {"help", "config"})
    if unknown:
        raise ValueError(f"Unknown YAML fields: {sorted(unknown)}")
    for name, value in values.items():
        action = actions[name]
        if value is None and action.default is not None:
            raise ValueError(f"{name} cannot be null")
        if isinstance(action.default, str) and not isinstance(value, str):
            raise ValueError(f"{name} must be a YAML string")
        if isinstance(
            action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)
        ) and not isinstance(value, bool):
            raise ValueError(f"{name} must be a YAML boolean")
        if action.type is not None and value is not None:
            if action.type in (int, float) and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (action.type is int and not isinstance(value, int))
            ):
                raise ValueError(f"{name} must be a YAML {action.type.__name__}")
            try:
                values[name] = action.type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid YAML value for {name}: {value}") from exc
        if action.choices and values[name] is not None and values[name] not in action.choices:
            raise ValueError(f"Invalid YAML choice for {name}: {value}")
    parser.set_defaults(**values)


def parse_arg(argv=None, prepare=True, training=True):
    # Resolve the family before constructing its argument surface. This keeps
    # TiTok-only and continuous-only settings from being silently ignored.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", default="")
    selector.add_argument("--prior_family", default=None)
    initial, _ = selector.parse_known_args(argv)
    values = load_yaml_config(initial.config)
    family = resolve_family(values, initial.prior_family)
    p = build_parser(family)
    apply_yaml_defaults(p, values)
    args = p.parse_args(argv)
    if args.prior_family != "titok":
        if args.prior_family == "imf" and (args.cfg_t_min is None) != (args.cfg_t_max is None):
            raise ValueError("CFG interval overrides require both endpoints")
        resolve_continuous_defaults(args)
    validate_args(args, training=training)
    args.objective = "wo_ps" if args.wo_ps else "w_ps"
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    set_run_name(args)
    if prepare:
        prepare_run(args)
    return args


def set_run_name(args):
    if not args.saver_name_pre:
        if args.prior_family == "titok":
            args.saver_name_pre = f"lpgan_{args.tokenizer_name}_{args.objective}"
        else:
            name = args.model_type if args.prior_family == "imf" else args.prior_family
            cfg = "cfg" if args.use_cfg else "nocfg"
            paired = f"_eta{args.supervision_noise_level:g}" if not args.wo_ps else ""
            paired_cfg = (
                ("_pairedcfg" if args.supervision_use_cfg else "_pairednocfg")
                if not args.wo_ps
                else ""
            )
            args.saver_name_pre = f"lpgan_{name}_{args.objective}{paired}_{cfg}{paired_cfg}"


def prepare_run(args):
    for path in (args.checkpoint_dir, args.results_dir, args.yaml_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", 0)) == 0:
        dest = Path(args.yaml_dir) / f"{args.saver_name_pre}.json"
        dest.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    if args.prior_family == "titok":
        # Preserve the token-pool experiment's RNG/backend configuration.
        import torch

        torch.manual_seed(args.seed)
    else:
        set_seed(args.seed)


def validate_args(args, training=True):
    import math

    for field in ("batch_size", "epochs", "save_interval", "log_interval"):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive")
    if args.workers < 0 or args.debug_steps < 0:
        raise ValueError("workers and debug_steps must be nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    for field in (
        "lr",
        "disc_lr",
        "weight_decay",
        "disc_weight",
        "lecam_loss_weight",
        "disc_cr_loss_weight",
    ):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{field} must be finite and nonnegative")
    if args.prior_family == "titok":
        if args.wo_ps:
            raise ValueError(
                "LP-GANs w/o PS is validated for continuous priors; TiTok retains its paired experimental objective"
            )
        if training and (not args.real_latent_pool or not args.generated_latent_pool):
            raise ValueError(
                "TiTok training requires --real_latent_pool and --generated_latent_pool"
            )
        if not math.isfinite(args.lr_decay) or not 0 < args.lr_decay <= 1:
            raise ValueError("--lr_decay must lie in (0,1]")
        for field in ("min_lr", "pairwise_weight"):
            if not math.isfinite(getattr(args, field)) or getattr(args, field) < 0:
                raise ValueError(f"--{field} must be finite and nonnegative")
        if args.save_interval != 1:
            raise ValueError(
                "TiTok retains the experimental checkpoint at every epoch (--save_interval 1)"
            )
        return
    if args.num_steps <= 0 or args.supervision_num_steps <= 0:
        raise ValueError("Sampling step counts must be positive")
    if not 0 <= args.supervision_noise_level <= 1:
        raise ValueError("Paired noise level must lie in [0,1]")
    if args.prior_family == "imf":
        if (
            not math.isfinite(args.cfg_omega)
            or args.cfg_omega < 1
            or not 0 <= args.cfg_t_min <= args.cfg_t_max <= 1
        ):
            raise ValueError("Invalid iMF CFG strength or interval")
    elif args.num_steps != 1 or args.supervision_num_steps != 1:
        raise ValueError("External-prior adapters use the validated one-step protocol")


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
