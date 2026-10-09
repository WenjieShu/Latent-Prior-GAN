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

from utils.decoder_checkpoint import load_decoder_checkpoint
from utils.sampling import indexed_inputs, pack_images, to_uint8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior_family", choices=["imf", "dmf", "meanflow", "titok"], default=None)
    p.add_argument("--config", default="", help="continuous-prior training YAML")
    for field in ("model_type", "pretrained_imf_pytorch", "prior_repo_path", "prior_checkpoint", "prior_vae_checkpoint", "prior_dtype", "vae_path"):
        p.add_argument("--" + field, default=None)
    for field in ("cfg_omega", "cfg_t_min", "cfg_t_max"):
        p.add_argument("--" + field, type=float, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--use_cfg", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--generator_name", choices=["titok-L32", "titok-B64"], default="titok-L32")
    p.add_argument("--generator_ckpt_path", default="")
    p.add_argument("--tokenizer_ckpt_path", default="")
    p.add_argument("--decoder_checkpoint", default="", help="empty evaluates the pretrained decoder")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32, help="per GPU")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", required=True, help="new directory; existing sample outputs are refused")
    args = p.parse_args(argv)
    if args.num_samples <= 0 or args.batch_size <= 0:
        p.error("num_samples and batch_size must be positive")
    if args.prior_family == "titok":
        if args.config or any(getattr(args, k) is not None for k in ("model_type", "pretrained_imf_pytorch", "prior_repo_path", "prior_checkpoint", "prior_vae_checkpoint", "prior_dtype", "vae_path", "cfg_t_min", "cfg_t_max")):
            p.error("TiTok uses its generator/tokenizer options, without continuous-prior configuration")
        from models.titok import registry
        steps = int(registry.config("generator", args.generator_name).model.generator.num_steps)
        if args.num_steps is not None and args.num_steps != steps:
            p.error(f"{args.generator_name} uses the registered {steps}-step protocol")
        args.num_steps = steps
        args.seed = 3407 if args.seed is None else args.seed
        args.use_cfg = True if args.use_cfg is None else args.use_cfg
        args.cfg_omega = float(registry.config("generator", args.generator_name).model.generator.guidance_scale) if args.cfg_omega is None else args.cfg_omega
        if not np.isfinite(args.cfg_omega) or args.cfg_omega < 0:
            p.error("Invalid TiTok guidance scale")
        args.model_name = args.generator_name
        args.resolution = 256
    else:
        import config
        cli = ["--config", args.config] if args.config else []
        for key in ("prior_family", "model_type", "pretrained_imf_pytorch", "prior_repo_path", "prior_checkpoint", "prior_vae_checkpoint", "prior_dtype", "vae_path", "cfg_omega", "cfg_t_min", "cfg_t_max", "num_steps", "seed"):
            if getattr(args, key) is not None:
                cli += ["--" + key, str(getattr(args, key))]
        if args.use_cfg is not None:
            cli += ["--use_cfg" if args.use_cfg else "--no-use_cfg"]
        resolved = config.parse_arg(cli, prepare=False)
        for key, value in vars(resolved).items():
            if key not in ("batch_size", "config"):
                setattr(args, key, value)
        args.model_name = args.model_type if args.prior_family == "imf" else args.prior_family
    if args.seed < 0 or args.seed >= 2 ** 32:
        p.error("seed must be in [0, 2**32)")
    return args


def build_models(args, device):
    if args.prior_family == "titok":
        from models.discrete_generator import Generator
        from models.discrete_tokenizer import Tokenizer
        from models.titok import registry
        tokenizer_name = registry.pair_for(args.generator_name)
        tokenizer = Tokenizer(tokenizer_name, args.tokenizer_ckpt_path).to(device)
        prior = Generator(args.generator_name, args.generator_ckpt_path, args.resolution, args.cfg_omega).to(device)
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
        model = LatentPriorGenerator(args).to(device).eval().requires_grad_(False)
        if args.decoder_checkpoint:
            load_decoder_checkpoint(model.tokenizer, args.decoder_checkpoint, args.prior_family, args.model_type)
        def sample(labels, seeds):
            latents = model.diffusion.generate(len(labels), labels=labels, num_steps=args.num_steps, use_cfg=args.use_cfg, device=device, seeds=seeds)
            return model.decode(latents)
        if args.prior_family == "imf":
            metadata = {"omega": args.cfg_omega if args.use_cfg else 1.0, "t_min": args.cfg_t_min if args.use_cfg else 0.0, "t_max": args.cfg_t_max if args.use_cfg else 1.0}
        elif args.prior_family == "dmf":
            metadata = dict(model.diffusion.defaults)
        else:
            from omegaconf import OmegaConf
            metadata = OmegaConf.to_container(model.diffusion.sampling, resolve=True)
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
        (out / "manifest.json").write_text(json.dumps({**vars(args), "world_size": world_size, "status": "generating", "seed_rule": "index XOR seed", "label_rule": "index modulo 1000"}, indent=2) + "\n")
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
