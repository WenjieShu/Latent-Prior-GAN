"""Precompute center-cropped ImageNet tokens with source-image pairing metadata."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from tqdm import tqdm

from data.dataloader import build_eval_transform
from torchvision.datasets import ImageFolder
from models.discrete_tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", default="ImageNet", choices=["ImageNet"])
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--tokenizer_ckpt_path", default="")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=0,
                        help="maximum samples; 0 uses the full split")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--indices_path", default="", help="NPY of source ImageFolder indices")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.resolution != 256 or args.num_samples < 0:
        raise ValueError("Invalid batch/workers/sample count or TiTok resolution")
    if not args.out.endswith(".npz") or os.path.exists(args.out):
        raise ValueError("--out must be a new .npz file")
    if not torch.cuda.is_available():
        raise RuntimeError("AR token precomputation requires CUDA")
    distributed = args.distributed or "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")
    data_path = args.dataset_dir if os.path.basename(os.path.normpath(args.dataset_dir)) == args.split else os.path.join(args.dataset_dir, args.split)
    if not os.path.exists(data_path):
        data_path = os.path.join(args.dataset_dir, "val" if args.split == "val" else "train")
    full_dataset = ImageFolder(data_path, transform=build_eval_transform(args.resolution))
    if args.indices_path:
        image_indices = np.load(args.indices_path)
        if image_indices.ndim != 1 or not np.issubdtype(image_indices.dtype, np.integer):
            raise ValueError("Source indices must be a one-dimensional integer array")
        if len(image_indices) == 0 or image_indices.min() < 0 or image_indices.max() >= len(full_dataset):
            raise ValueError("Source indices outside ImageFolder")
        if len(np.unique(image_indices)) != len(image_indices):
            raise ValueError("Source indices contain duplicates")
        if args.num_samples > 0 and args.num_samples != len(image_indices):
            raise ValueError("num_samples differs from source index count")
    else:
        total = len(full_dataset) if args.num_samples <= 0 else min(args.num_samples, len(full_dataset))
        image_indices = np.arange(total, dtype=np.int64)
    total = len(image_indices)
    if total < world_size:
        raise ValueError("Each rank requires at least one image")
    indices = list(range(rank, total, world_size))
    dataset_loader = DataLoader(
        Subset(full_dataset, image_indices[indices].tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        drop_last=False, persistent_workers=args.workers > 0)
    tokenizer = Tokenizer(args.tokenizer_name, args.tokenizer_ckpt_path).to(device)
    tokenizer.eval_mode()

    token_batches = []
    label_batches = []
    with torch.no_grad():
        seen = 0
        for images, labels in tqdm(dataset_loader, desc=f"tokenize rank {rank}", disable=rank != 0):
            tokens = tokenizer.extract_latents(images.to(device, non_blocking=True))
            token_batches.append(tokens.cpu().numpy())
            label_batches.append(labels.numpy())
            seen += len(labels)

    tokens_local = np.concatenate(token_batches, axis=0)
    labels_local = np.concatenate(label_batches, axis=0).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    shard = f"{os.path.splitext(args.out)[0]}.rank{rank}.npz"
    np.savez(shard, indices=np.asarray(indices, dtype=np.int64), tokens=tokens_local, labels=labels_local)
    if distributed:
        dist.barrier()
    if rank == 0:
        shards = [np.load(f"{os.path.splitext(args.out)[0]}.rank{r}.npz") for r in range(world_size)]
        tokens = np.empty((total, tokens_local.shape[1]), dtype=tokens_local.dtype)
        labels = np.empty(total, dtype=np.int16)
        for data in shards:
            tokens[data["indices"]] = data["tokens"]
            labels[data["indices"]] = data["labels"]
        sample_paths = np.asarray([
            os.path.relpath(full_dataset.samples[int(index)][0], data_path)
            for index in image_indices
        ])
        np.savez(args.out, tokens=tokens, labels=labels,
             image_indices=image_indices, sample_paths=sample_paths,
             preprocessing="adm_center_crop_v1", split=args.split,
             tokenizer_name=args.tokenizer_name,
             tokenizer_ckpt_path=args.tokenizer_ckpt_path,
             resolution=args.resolution)
        for data in shards:
            data.close()
        for r in range(world_size):
            os.remove(f"{os.path.splitext(args.out)[0]}.rank{r}.npz")
        print(f"Saved {args.out}: tokens={tokens.shape}, labels={labels.shape}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
