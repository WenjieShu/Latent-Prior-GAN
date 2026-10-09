"""Sample a pool of generation-time latents into one .npz: tokens (N, T) int16/int32, labels (N,) int16.
Each rank streams its strided shard to a memmap with a progress file (resumable); rank 0 merges.

    python precompute_titok_generated_tokens.py --generator_name titok-L32 \
        --num_samples 1280000 --label_pool pools/real.npz --deterministic --out pools/generated.npz
"""

import argparse
import os
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from models.discrete_generator import Generator
from models.titok import registry
from utils.pool_resume import validate_shard_resume


def parse_args():
    p = argparse.ArgumentParser(description="Sample a pool of generation-time latents")
    p.add_argument("--generator_name", choices=["titok-L32", "titok-B64"], required=True)
    p.add_argument("--generator_ckpt_path", type=str, default="")
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256,
                   help="per-rank batch size")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--label_pool", type=str, default="",
                   help="npz containing labels; preserves pairing with a real token pool")
    p.add_argument("--cfg_omega", type=float, default=None,
                   help="guidance scale; None = registry value")
    p.add_argument("--use_cfg", action="store_true", default=True)
    p.add_argument("--no_cfg", dest="use_cfg", action="store_false")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; --deterministic uses seed ^ index per sample")
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="seed each sample separately")
    p.add_argument("--restart", action="store_true", default=False,
                   help="discard existing shards instead of resuming")
    return p.parse_args()


def interleaved_labels(num_samples, num_classes):
    """-> (num_samples,) int16; index i has class i % num_classes."""
    return (np.arange(num_samples, dtype=np.int64) % num_classes).astype(np.int16)


def shard_paths(out, rank):
    base = os.path.splitext(out)[0]
    return f"{base}.rank{rank}.npy", f"{base}.rank{rank}.progress"


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_samples <= 0 or args.num_classes != 1000 or not 0 <= args.seed < 2**32:
        raise ValueError("Require positive batch/sample counts, ImageNet 1000 classes, and a uint32 seed")
    if not args.out.endswith(".npz") or Path(args.out).exists():
        raise ValueError("--out must be a new .npz file")
    if not torch.cuda.is_available():
        raise RuntimeError("TiTok token generation requires CUDA")
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)

    reg = registry
    if not args.generator_ckpt_path:
        args.generator_ckpt_path = reg.ckpt_path(
            "generator", args.generator_name, args.resolution)

    token_length = reg.token_length(args.resolution, name=args.generator_name)
    dtype = np.int16 if reg.CODEBOOK_SIZE <= np.iinfo(np.int16).max else np.int32
    # per-rank seed offset
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if args.label_pool:
        label_data = np.load(args.label_pool, mmap_mode="r")
        labels = np.asarray(label_data["labels"])
        if len(labels) != args.num_samples:
            raise ValueError("--label_pool labels length must equal --num_samples")
        labels = labels.astype(np.int16, copy=False)
        if "image_indices" not in label_data:
            raise ValueError("--label_pool must contain source image_indices")
        image_indices = np.asarray(label_data["image_indices"])
        if image_indices.shape != (args.num_samples,):
            raise ValueError("--label_pool image_indices shape mismatch")
        if str(label_data["tokenizer_name"].item()) != reg.pair_for(args.generator_name):
            raise ValueError("Generator/tokenizer pairing differs")
        label_data.close()
    else:
        labels = interleaved_labels(args.num_samples, args.num_classes)
    if labels.shape != (args.num_samples,) or np.any(labels < 0) or np.any(labels >= 1000):
        raise ValueError("Invalid pool labels")

    # strided shard: rank r owns indices r, r+world_size, ...
    my_idx = np.arange(rank, args.num_samples, world_size)
    tok_path, prog_path = shard_paths(args.out, rank)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    checkpoint_hash = [None]
    if rank == 0:
        digest = hashlib.sha256()
        with open(args.generator_ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_hash[0] = digest.hexdigest()
    if distributed:
        dist.broadcast_object_list(checkpoint_hash, src=0, device=device)
    signature = {"generator_name": args.generator_name, "checkpoint_sha256": checkpoint_hash[0],
                 "world_size": world_size, "num_samples": args.num_samples, "batch_size": args.batch_size,
                 "cfg_omega": args.cfg_omega, "use_cfg": args.use_cfg, "seed": args.seed,
                 "deterministic": args.deterministic, "resolution": args.resolution,
                 "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                 "indices_sha256": hashlib.sha256(image_indices.tobytes()).hexdigest() if args.label_pool else None}
    meta_path = tok_path + ".json"
    done = validate_shard_resume(meta_path, signature, tok_path, prog_path, len(my_idx), args.deterministic, args.restart)
    if os.path.exists(tok_path):
        existing = np.lib.format.open_memmap(tok_path, mode="r")
        if existing.shape != (len(my_idx), token_length) or existing.dtype != dtype:
            raise ValueError("Shard shape/dtype differs; use --restart")
        del existing
    mode = "r+" if done else "w+"
    tokens_local = np.lib.format.open_memmap(
        tok_path, mode=mode, dtype=dtype, shape=(len(my_idx), token_length))
    if done and rank == 0:
        print(f"[resume] {done}/{len(my_idx)} per rank already sampled", flush=True)

    generator = Generator(args.generator_name, args.generator_ckpt_path, args.resolution,
                          cfg_omega=args.cfg_omega).to(device)
    generator.eval_mode()

    it = range(done, len(my_idx), args.batch_size)
    for i in tqdm(it, desc="sample", disable=rank != 0):
        idx = my_idx[i:i + args.batch_size]
        y = torch.from_numpy(labels[idx].astype(np.int64)).to(device)
        seeds = (args.seed ^ idx) if args.deterministic else None
        out = generator.standard_generation(len(idx), labels=y, use_cfg=args.use_cfg,
                                            device=device, seeds=seeds)
        tokens_local[i:i + len(idx)] = out.cpu().numpy().astype(dtype)
        tokens_local.flush()
        with open(prog_path, "w") as f:
            f.write(str(i + len(idx)))

    if distributed:
        dist.barrier()

    if rank == 0:
        tokens = np.empty((args.num_samples, token_length), dtype=dtype)
        for r in range(world_size):
            path, _ = shard_paths(args.out, r)
            shard = np.lib.format.open_memmap(path, mode="r")
            tokens[np.arange(r, args.num_samples, world_size)] = shard
            del shard
        metadata = {"image_indices": image_indices} if args.label_pool else {}
        np.savez(args.out, tokens=tokens, labels=labels, **metadata,
                 generator_name=args.generator_name,
                 generator_ckpt_path=args.generator_ckpt_path,
                 resolution=args.resolution, use_cfg=args.use_cfg,
                 cfg_omega=generator.model.cfg_omega, seed=args.seed,
                 deterministic=args.deterministic, checkpoint_sha256=checkpoint_hash[0])
        print(f"[pool] {args.out}  tokens {tokens.shape} {tokens.dtype}  "
              f"{tokens.nbytes / 1024 ** 2:.0f} MB", flush=True)
        for r in range(world_size):
            tok_file, prog_file = shard_paths(args.out, r)
            for p in (tok_file, prog_file, tok_file + ".json"):
                if os.path.exists(p):
                    os.remove(p)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
