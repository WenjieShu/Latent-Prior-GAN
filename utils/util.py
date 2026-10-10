"""
General utilities.

Follows the structure of VQ-Transplant/VAR/utils/util.py
"""

import os
import shutil
import sys
from collections import defaultdict

import torch


class Pack(dict):
    """
    A dictionary that allows attribute-style access.
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class Logger:
    """
    Logger that writes to both stdout and file.
    """

    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.stdout
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.log = open(self.log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class LossManager:
    """
    Manager for tracking and averaging losses.
    """

    def __init__(self):
        self.losses = defaultdict(list)

    def add_loss(self, pack):
        """Add losses from a Pack object."""
        for key, value in pack.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.losses[key].append(value)

    def clear(self):
        """Clear all tracked losses."""
        self.losses = defaultdict(list)

    def pprint(self, window=50, prefix=""):
        """Pretty print recent losses."""
        parts = [prefix]
        for key, values in self.losses.items():
            recent = values[-window:] if len(values) > window else values
            avg = sum(recent) / len(recent) if recent else 0
            parts.append(f"{key}: {avg:.4f}")
        return " | ".join(parts)

    def get_avg(self, key, window=None):
        """Get average of a specific loss."""
        values = self.losses.get(key, [])
        if window is not None:
            values = values[-window:]
        return sum(values) / len(values) if values else 0


def save_checkpoint(state, is_best, filename):
    """
    Save model checkpoint.

    Args:
        state: State dict containing model, optimizer, etc.
        is_best: Whether this is the best checkpoint
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")

    if is_best:
        best_filename = filename.replace(".pth.tar", "_best.pth.tar")
        shutil.copyfile(filename, best_filename)
        print(f"Saved best checkpoint to {best_filename}")
