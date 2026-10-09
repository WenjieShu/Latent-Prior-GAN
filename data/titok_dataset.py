import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from data.dataloader import build_eval_transform


class PairedARLatentDataset(Dataset):
    """Index-aligned tokens plus the image paired with the real tokens."""

    def __init__(self, real_pool_path, generated_pool_path, dataset_dir, resolution=256):
        real = np.load(real_pool_path, mmap_mode="r")
        generated = np.load(generated_pool_path, mmap_mode="r")
        for description, pool in (("real", real), ("generated", generated)):
            for key in ("tokens", "labels", "image_indices"):
                if key not in pool:
                    raise ValueError(f"{description} token pool missing {key}; regenerate the pool")
            tokens = pool["tokens"]
            if tokens.ndim != 2 or not len(tokens) or not np.issubdtype(tokens.dtype, np.integer):
                raise ValueError(f"Invalid {description} tokens")
            if tokens.min() < 0 or tokens.max() >= 4096:
                raise ValueError(f"{description} token outside TiTok codebook")
            for key in ("labels", "image_indices"):
                if pool[key].shape != (len(tokens),) or not np.issubdtype(pool[key].dtype, np.integer):
                    raise ValueError(f"Invalid {description} {key}")
        if real["tokens"].shape != generated["tokens"].shape:
            raise ValueError("Real/generated token shapes differ")
        if len(real["tokens"]) != len(generated["tokens"]):
            raise ValueError("real and generated pools must have the same length")
        if not np.array_equal(real["labels"], generated["labels"]):
            raise ValueError("real and generated pools must have identical labels")
        self.real_tokens = real["tokens"]
        self.generated_tokens = generated["tokens"]
        self.labels = real["labels"]
        if "preprocessing" not in real or real["preprocessing"].item() != "adm_center_crop_v1":
            raise ValueError("Real tokens need verified center-crop metadata; regenerate legacy pools")
        if int(real["resolution"]) != resolution:
            raise ValueError("Token pool and target image resolutions differ")
        if "resolution" in generated and int(generated["resolution"]) != resolution:
            raise ValueError("Generated token resolution differs")
        if "split" in real and real["split"].item() != "train":
            raise ValueError("Training requires tokens from the train split")
        if "generator_name" in generated and "tokenizer_name" in real:
            from models.titok.registry import pair_for
            if pair_for(generated["generator_name"].item()) != real["tokenizer_name"].item():
                raise ValueError("Generator/tokenizer pairing differs")
        self.indices = real["image_indices"]
        self.sample_paths = real["sample_paths"]
        if not np.array_equal(self.indices, generated["image_indices"]):
            raise ValueError("Real and generated pools must use identical source indices")
        data_path = os.path.normpath(dataset_dir)
        if os.path.basename(data_path) != "train":
            data_path = os.path.join(data_path, "train")
        self.image_dataset = ImageFolder(
            data_path, transform=build_eval_transform(resolution),
        )
        if len(self.indices) != len(self.real_tokens) or len(self.sample_paths) != len(self.indices):
            raise ValueError("Image index/path metadata length differs from pool size")
        if self.indices.min() < 0 or self.indices.max() >= len(self.image_dataset):
            raise ValueError("Source image index outside ImageNet train split")
        for pool_index, image_index in enumerate(self.indices):
            path, label = self.image_dataset.samples[int(image_index)]
            if label != int(self.labels[pool_index]):
                raise ValueError(f"Image/token label mismatch at pool index {pool_index}")
            if os.path.relpath(path, data_path) != self.sample_paths[pool_index]:
                raise ValueError(f"Image/token path mismatch at pool index {pool_index}")
        real.close()
        generated.close()

    def __len__(self):
        return len(self.real_tokens)

    def __getitem__(self, index):
        image, image_label = self.image_dataset[int(self.indices[index])]
        if int(image_label) != int(self.labels[index]):
            raise ValueError(f"Image/token label mismatch at index {index}")
        return (
            torch.from_numpy(self.real_tokens[index].astype(np.int64)),
            torch.from_numpy(self.generated_tokens[index].astype(np.int64)),
            image,
        )
