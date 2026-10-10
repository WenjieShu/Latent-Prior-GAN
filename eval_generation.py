"""Generate indexed ImageNet samples from a frozen prior and an optional LP-GAN decoder."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

import config
from utils.decoder_checkpoint import load_decoder_checkpoint
from utils.sampling import indexed_inputs, pack_images, to_uint8

CONTINUOUS_OPTIONS = (
    "model_type",
    "pretrained_imf_pytorch",
    "prior_repo_path",
    "prior_checkpoint",
    "prior_vae_checkpoint",
    "prior_dtype",
    "vae_path",
    "cfg_omega",
    "cfg_t_min",
    "cfg_t_max",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior_family", choices=["imf", "dmf", "meanflow", "titok"], default=None)
    p.add_argument(
        "--config",
        default="",
        help="model-specific training YAML; training-only settings do not control evaluation",
    )
    for field in (
        "model_type",
        "pretrained_imf_pytorch",
        "prior_repo_path",
        "prior_checkpoint",
        "prior_vae_checkpoint",
        "prior_dtype",
        "vae_path",
    ):
        p.add_argument("--" + field, default=None)
    for field in ("cfg_omega", "cfg_t_min", "cfg_t_max"):
        p.add_argument("--" + field, type=float, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--use_cfg", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--generator_name", choices=["titok-L32", "titok-B64"], default=None)
    p.add_argument("--generator_ckpt_path", default="")
    p.add_argument("--tokenizer_ckpt_path", default="")
    p.add_argument(
        "--decoder_checkpoint", default="", help="empty evaluates the pretrained decoder"
    )
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32, help="per GPU")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--out_dir", required=True, help="new directory; existing sample outputs are refused"
    )
    args = p.parse_args(argv)
    if args.num_samples <= 0 or args.batch_size <= 0:
        p.error("num_samples and batch_size must be positive")
    values = config.load_yaml_config(args.config)
    args.prior_family = config.resolve_family(values, args.prior_family)
    cli = ["--config", args.config, "--prior_family", args.prior_family]
    if args.seed is not None:
        cli += ["--seed", str(args.seed)]
    if args.prior_family == "titok":
        if any(getattr(args, key) is not None for key in CONTINUOUS_OPTIONS if key != "cfg_omega"):
            p.error(
                "TiTok uses generator/tokenizer options; continuous-prior options are unsupported"
            )
        from models.titok import registry

        if args.generator_name:
            cli += ["--tokenizer_name", registry.pair_for(args.generator_name)]
        if args.tokenizer_ckpt_path:
            cli += ["--tokenizer_ckpt_path", args.tokenizer_ckpt_path]
        resolved = config.parse_arg(cli, prepare=False, training=False)
        args.generator_name = args.generator_name or registry.generator_for(resolved.tokenizer_name)
        args.tokenizer_ckpt_path = resolved.tokenizer_ckpt_path
        steps = int(registry.config("generator", args.generator_name).model.generator.num_steps)
        if args.num_steps is not None and args.num_steps != steps:
            p.error(f"{args.generator_name} uses the registered {steps}-step protocol")
        args.num_steps = steps
        args.seed = resolved.seed
        args.use_cfg = True if args.use_cfg is None else args.use_cfg
        args.cfg_omega = (
            float(registry.config("generator", args.generator_name).model.generator.guidance_scale)
            if args.cfg_omega is None
            else args.cfg_omega
        )
        if not np.isfinite(args.cfg_omega) or args.cfg_omega < 0:
            p.error("Invalid TiTok guidance scale")
        args.model_name = args.generator_name
        args.resolution = resolved.resolution
    else:
        if args.generator_name or args.generator_ckpt_path or args.tokenizer_ckpt_path:
            p.error(
                "Generator/tokenizer checkpoint options are specific to TiTok; use --vae_path for continuous priors"
            )
        for key in (*CONTINUOUS_OPTIONS, "num_steps"):
            if getattr(args, key) is not None:
                cli += ["--" + key, str(getattr(args, key))]
        if args.use_cfg is not None:
            cli += ["--use_cfg" if args.use_cfg else "--no-use_cfg"]
        resolved = config.parse_arg(cli, prepare=False, training=False)
        # Import only model/sampling settings. Training initialization, optimizer,
        # output paths, and the PS objective must not affect decoder evaluation.
        for key in (
            *CONTINUOUS_OPTIONS,
            "num_steps",
            "use_cfg",
            "seed",
            "resolution",
            "vae_type",
            "latent_size",
        ):
            if hasattr(resolved, key):
                setattr(args, key, getattr(resolved, key))
        args.decoder_init = "pretrained"
        args.model_name = args.model_type if args.prior_family == "imf" else args.prior_family
    if args.seed < 0 or args.seed >= 2**32:
        p.error("seed must be in [0, 2**32)")
    return args


def build_models(args, device):
    if args.prior_family == "titok":
        from models.discrete_generator import Generator
        from models.discrete_tokenizer import Tokenizer
        from models.titok import registry

        tokenizer_name = registry.pair_for(args.generator_name)
        tokenizer = Tokenizer(tokenizer_name, args.tokenizer_ckpt_path).to(device)
        prior = Generator(
            args.generator_name, args.generator_ckpt_path, args.resolution, args.cfg_omega
        ).to(device)
        prior.eval_mode()
        if args.decoder_checkpoint:
            load_decoder_checkpoint(tokenizer, args.decoder_checkpoint, "titok", tokenizer_name)
        tokenizer.eval_mode()

        def sample(labels, seeds):
            tokens = prior.standard_generation(len(labels), labels, args.use_cfg, device, seeds)
            return tokenizer.latent_to_image(tokens)

        metadata = dict(prior.model.sampling)
        metadata["guidance_scale"] = args.cfg_omega if args.use_cfg else 0.0
    else:
        from models.model import LatentPriorGenerator

        model = LatentPriorGenerator(args).to(device)
        model.eval_mode()
        if args.decoder_checkpoint:
            load_decoder_checkpoint(
                model.tokenizer, args.decoder_checkpoint, args.prior_family, args.model_name
            )

        def sample(labels, seeds):
            return model.standard_generation(labels, seeds)

        metadata = model.sampling_metadata()
    return sample, metadata


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full generation requires CUDA; contract tests run on CPU")
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    out = Path(args.out_dir)
    # Every rank checks before rank zero creates outputs.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory: {out}")
    if distributed:
        dist.barrier()
    if rank == 0:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "world_size": world_size,
                    "status": "generating",
                    "seed_rule": "index XOR seed",
                    "label_rule": "index modulo 1000",
                },
                indent=2,
            )
            + "\n"
        )
    if distributed:
        dist.barrier()
    sample, sampling_metadata = build_models(args, device)
    if rank == 0:
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["native_sampling"] = sampling_metadata
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    indices, labels, seeds = indexed_inputs(args.num_samples, args.seed, rank, world_size)
    with torch.inference_mode():
        for start in tqdm(range(0, len(indices), args.batch_size), disable=rank != 0):
            batch = slice(start, start + args.batch_size)
            y = torch.from_numpy(labels[batch]).to(device)
            images = to_uint8(sample(y, torch.from_numpy(seeds[batch])))
            if len(images) != len(indices[batch]):
                raise ValueError("Generator returned an incorrect batch size")
            for idx, image in zip(indices[batch], images):
                Image.fromarray(image).save(out / "images" / f"{idx:06d}.png")
    if distributed:
        dist.barrier()
    if rank == 0:
        pack_images(out / "images", args.num_samples, args.resolution, out / "samples.npz")
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {args.num_samples} samples to {out / 'samples.npz'}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
