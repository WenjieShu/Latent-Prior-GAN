"""Shared rank and process-group utilities for continuous and TiTok training."""

import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed_mode(args, *, allow_single=False, timeout=None):
    """Resolve torchrun/Slurm ranks; retain each experiment's timeout policy."""
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostname", os.environ["SLURM_NODELIST"]],
            text=True,
        ).splitlines()
        os.environ.setdefault("MASTER_ADDR", hostnames[0])
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if not allow_single:
            raise RuntimeError(
                "Launch training with torchrun --standalone --nproc_per_node=1 (or more GPUs)"
            )
        args.distributed = False
        args.rank, args.world_size, args.gpu = 0, 1, 0
        return
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.distributed = True
    args.dist_backend = getattr(args, "dist_backend", "nccl")
    torch.cuda.set_device(args.gpu)
    options = dict(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    if timeout is not None:
        options["timeout"] = timedelta(seconds=timeout)
    print(f"| distributed init (rank {args.rank}): gpu {args.gpu}", flush=True)
    dist.init_process_group(**options)
    dist.barrier()


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


def is_main_process():
    return get_rank() == 0


def finish_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
