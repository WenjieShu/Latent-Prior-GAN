"""Small training helpers shared by the entry points."""

import os
import random

import numpy as np
import torch


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable
