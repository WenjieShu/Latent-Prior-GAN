"""Indexed sampling and exact-count image packing shared across priors."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def indexed_inputs(count, seed, rank=0, world_size=1, num_classes=1000):
    if count <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid sample count or shard")
    indices = np.arange(rank, count, world_size, dtype=np.int64)
    return indices, indices % num_classes, indices ^ seed


def to_uint8(images):
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError("Expected finite NCHW RGB images")
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def pack_images(directory, count, resolution, destination):
    directory = Path(directory)
    expected = {f"{i:06d}.png" for i in range(count)}
    actual = {p.name for p in directory.glob("*.png")}
    if actual != expected:
        raise ValueError(f"Sample count mismatch: missing={len(expected - actual)}, unexpected={len(actual - expected)}")
    # Stream to a disk-backed array instead of retaining 50K images in RAM.
    scratch = Path(str(destination) + ".packing.npy")
    images = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.uint8, shape=(count, resolution, resolution, 3))
    try:
        for i in range(count):
            with Image.open(directory / f"{i:06d}.png") as img:
                array = np.asarray(img)
            if array.shape != (resolution, resolution, 3) or array.dtype != np.uint8:
                raise ValueError(f"Invalid generated image: {i}")
            images[i] = array
        images.flush()
        np.savez(destination, arr_0=images)
    finally:
        del images
        scratch.unlink(missing_ok=True)
