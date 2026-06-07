"""Latent Prior GAN training entry point."""

import argparse
import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataloader import build_dataloader
from models.diffusion_pytorch import DiffusionModelPyTorch
from models.latent_prior_gan_model import LatentPriorGAN
from utils.checkpointing import load_tokenizer_state, tokenizer_prefixed_state, tokenizer_state
from utils.config import load_config, save_config
from utils.distributed import init_distributed_mode, is_main_process
from utils.training import enable_tf32, seed_everything, unwrap
from utils.util import Logger, LossManager, save_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

enable_tf32()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Prior GAN: decoder adaptation on generated iMF latents."
    )
    parser.add_argument("--config", default="", type=str, help="optional YAML config")

    parser.add_argument("--dataset_dir", required=False, default="/mnt/nfs/wenjie/dataset/imagenet_idx")
    parser.add_argument("--dataset_name", default="ImageNet", choices=["ImageNet"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--model_type", default="iMF-XL-2", choices=["iMF-B-2", "iMF-M-2", "iMF-L-2", "iMF-XL-2"])
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--vae_type", default="mse", choices=["mse", "ema"])
    parser.add_argument("--pretrained_imf_pytorch", required=False, default="")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--cfg_omega", type=float, default=None)
    parser.add_argument("--cfg_t_min", type=float, default=None)
    parser.add_argument("--cfg_t_max", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume", default="", type=str)

    parser.add_argument("--disc_weight", type=float, default=0.7)
    parser.add_argument("--disc_start_epoch", type=int, default=0)
    parser.add_argument("--lecam_loss_weight", type=float, default=0.001)
    parser.add_argument("--disc_cr_loss_weight", type=float, default=4.0)
    parser.add_argument("--diff_aug_policy", default="color,translation,cutout_0.2")
    parser.add_argument("--diff_aug_prob", type=float, default=0.5)
    parser.add_argument("--diff_aug_cr_policy", default="color,translation,cutout_0.5")

    parser.add_argument("--checkpoint_dir", default="/mnt/nfs/wenjie/Latent-Prior-GAN/checkpoints")
    parser.add_argument("--results_dir", default="/mnt/nfs/wenjie/Latent-Prior-GAN/results")
    parser.add_argument("--yaml_dir", default="/mnt/nfs/wenjie/Latent-Prior-GAN/yaml")
    parser.add_argument("--saver_name_pre", default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="Latent-Prior-GAN")
    parser.add_argument("--wandb_run_name", default="")

    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--local-rank", default=-1, type=int)

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")

    config = load_config(args.config)
    if config:
        defaults = {k: v for k, v in config.items() if hasattr(args, k)}
        parser.set_defaults(**defaults)
        args = parser.parse_args()

    if not args.pretrained_imf_pytorch:
        args.pretrained_imf_pytorch = (
            f"/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/{args.model_type}.pt"
        )
    if not args.saver_name_pre:
        cfg_tag = "cfg" if args.use_cfg else "nocfg"
        args.saver_name_pre = f"latent_prior_gan_{args.model_type}_nt1_gan_only_{cfg_tag}"
    return args


def fill_cfg_defaults(args):
    cfg = DiffusionModelPyTorch.CFG_PARAMS[args.model_type]
    if args.cfg_omega is None:
        args.cfg_omega = cfg["omega"]
    if args.cfg_t_min is None:
        args.cfg_t_min = cfg["t_min"]
    if args.cfg_t_max is None:
        args.cfg_t_max = cfg["t_max"]


def main():
    args = parse_args()
    fill_cfg_defaults(args)
    seed_everything(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    init_distributed_mode(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Latent Prior GAN training requires CUDA.")
    if not getattr(args, "distributed", False):
        args.gpu = 0
        args.rank = 0
        args.world_size = 1
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    if is_main_process():
        sysout_path = os.path.join(args.results_dir, "logs")
        os.makedirs(sysout_path, exist_ok=True)
        import sys
        sys.stdout = Logger(sysout_path, args.saver_name_pre)
        print("=" * 72)
        print("Latent Prior GAN decoder post-training")
        print("=" * 72)
        print("Objective: GAN only on x_fake = D_trainable(G_iMF(epsilon, y)); nt=1.0")
        print("No paired reconstruction, no LPIPS, no Gram loss.")
        for key, value in sorted(vars(args).items()):
            print(f"{key}: {value}")
        print("=" * 72)
        print(f"Saved config: {save_config(args)}")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.saver_name_pre,
                config=vars(args),
            )

    train_loader, train_sampler, train_size = build_dataloader(args, split="train")
    if is_main_process():
        print(f"Train size: {train_size}")

    start_epoch = 1
    global_step = 0
    checkpoint = None

    model = LatentPriorGAN(args).to(device)
    model.load_prior()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        load_tokenizer_state(model.tokenizer, checkpoint["decoder"])
        model.gan_loss.load_state_dict(checkpoint["loss"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if getattr(args, "distributed", False):
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    model.train()

    base_model = unwrap(model)
    decoder_params = base_model.decoder_parameters()
    disc_params = base_model.discriminator_parameters()
    optimizer_dec = torch.optim.AdamW(
        decoder_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    optimizer_disc = torch.optim.AdamW(
        disc_params, lr=args.disc_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    if checkpoint is not None:
        optimizer_dec.load_state_dict(checkpoint["optimizer_dec"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])

    train_meter = LossManager()
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        for step, (real_images, labels) in enumerate(train_loader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer_dec.zero_grad(set_to_none=True)
            optimizer_disc.zero_grad(set_to_none=True)
            g_loss, g_pack = model(mode="generator", labels=labels)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            optimizer_dec.step()

            disc_active = epoch >= args.disc_start_epoch
            if disc_active:
                optimizer_disc.zero_grad(set_to_none=True)
                d_loss, d_pack = model(
                    mode="discriminator",
                    real_images=real_images,
                    labels=labels,
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
                optimizer_disc.step()
                g_pack.update(d_pack)

            global_step += 1
            train_meter.add_loss(g_pack)

            if is_main_process() and step % args.log_interval == 0:
                msg = train_meter.pprint(
                    window=args.log_interval,
                    prefix=f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}]",
                )
                print(msg)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({f"train/{k}": train_meter.get_avg(k, args.log_interval)
                               for k in train_meter.losses}, step=global_step)

            if args.max_train_steps and step >= args.max_train_steps:
                if is_main_process():
                    print(f"Stopping epoch early after {args.max_train_steps} steps.")
                break

        if is_main_process():
            print(f"Epoch {epoch} finished in {time.time() - epoch_start:.1f}s")
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(
                    args.checkpoint_dir,
                    f"checkpoint-{args.saver_name_pre}-epoch{epoch:03d}.pth.tar",
                )
                save_checkpoint(
                    {
                        "decoder": tokenizer_state(base_model.tokenizer),
                        "model": tokenizer_prefixed_state(base_model.tokenizer),
                        "loss": base_model.gan_loss.state_dict(),
                        "optimizer_dec": optimizer_dec.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": vars(args),
                    },
                    is_best=False,
                    filename=ckpt_path,
                )

        train_meter.clear()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main_process():
        print("Latent Prior GAN training complete.")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
