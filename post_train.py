"""
Extension-GAR decoder post-training.

This entry point adapts only the SD-VAE decoder on pure iMF-generated latents.
It intentionally does not use paired reconstruction, LPIPS, or Gram losses.

Training objective:
    z_g = G_iMF(epsilon, y)          # frozen iMF generator, nt = 1.0
    x_fake = D_trainable(z_g)        # trainable SD-VAE decoder
    GAN(real ImageNet images, x_fake)
"""

import argparse
import json
import os
import random
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataloader import build_dataloader
from models.diffusion_pytorch import DiffusionModelPyTorch
from models.generation_gan_loss import GenerationGANLoss
from models.tokenizer import Tokenizer
from utils.distributed import init_distributed_mode
from utils.util import Logger, LossManager, save_checkpoint

try:
    import yaml
except ImportError:
    yaml = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def load_config_file(path):
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required when --config is used.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension-GAR: GAN-only decoder adaptation on generated iMF latents."
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

    parser.add_argument("--checkpoint_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/checkpoints/extension_gar")
    parser.add_argument("--results_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/results/extension_gar")
    parser.add_argument("--yaml_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/yaml/extension_gar")
    parser.add_argument("--saver_name_pre", default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="IMF-GAP-V2")
    parser.add_argument("--wandb_run_name", default="")

    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--local-rank", default=-1, type=int)

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")

    config = load_config_file(args.config)
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
        args.saver_name_pre = f"extension_gar_{args.model_type}_nt1_gan_only_{cfg_tag}"
    return args


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_args(args):
    os.makedirs(args.yaml_dir, exist_ok=True)
    path = os.path.join(args.yaml_dir, f"{args.saver_name_pre}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    return path


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def tokenizer_prefixed_state(decoder_module):
    tokenizer_state = clean_state_dict(decoder_module.tokenizer, "vae.decoder.module.", "vae.decoder.")
    return {f"tokenizer.{k}": v for k, v in tokenizer_state.items()}


def clean_state_dict(module, wrapped_prefix, clean_prefix):
    state = module.state_dict()
    return {
        (key.replace(wrapped_prefix, clean_prefix, 1) if key.startswith(wrapped_prefix) else key): value
        for key, value in state.items()
    }


def ddp_decoder_state_dict(decoder_module):
    return clean_state_dict(
        decoder_module,
        "tokenizer.vae.decoder.module.",
        "tokenizer.vae.decoder.",
    )


def load_ddp_decoder_state(decoder_module, state):
    decoder_is_wrapped = hasattr(decoder_module.tokenizer.vae.decoder, "module")
    remapped = {}
    for key, value in state.items():
        has_module_prefix = key.startswith("tokenizer.vae.decoder.module.")
        has_decoder_prefix = key.startswith("tokenizer.vae.decoder.")
        if decoder_is_wrapped and has_decoder_prefix and not has_module_prefix:
            key = key.replace("tokenizer.vae.decoder.", "tokenizer.vae.decoder.module.", 1)
        elif not decoder_is_wrapped and has_module_prefix:
            key = key.replace("tokenizer.vae.decoder.module.", "tokenizer.vae.decoder.", 1)
        remapped[key] = value
    decoder_module.load_state_dict(remapped, strict=True)


class GeneratedLatentDecoder(torch.nn.Module):
    """Trainable SD-VAE decoder applied to iMF-generated latents."""

    def __init__(self, args):
        super().__init__()
        self.tokenizer = Tokenizer(args)
        self.tokenizer.train_decoder()

    def forward(self, z_gen):
        return self.tokenizer.decode(z_gen).clamp(-1, 1)


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
        raise RuntimeError("Extension-GAR training requires CUDA.")
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
        print("Extension-GAR generation-GAN decoder post-training")
        print("=" * 72)
        print("Objective: GAN only on x_fake = D_trainable(G_iMF(epsilon, y)); nt=1.0")
        print("No paired reconstruction, no LPIPS, no Gram loss.")
        for key, value in sorted(vars(args).items()):
            print(f"{key}: {value}")
        print("=" * 72)
        print(f"Saved config: {save_args(args)}")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.saver_name_pre,
                config=vars(args),
            )

    train_loader, train_sampler, train_size = build_dataloader(args, split="train")
    if is_main_process():
        print(f"Train size: {train_size}")

    decoder = GeneratedLatentDecoder(args).to(device)
    set_requires_grad(decoder.tokenizer.vae, False)
    set_requires_grad(decoder.tokenizer.vae.decoder, True)
    decoder.tokenizer.vae.encoder.eval()
    decoder.tokenizer.vae.decoder.train()

    diffusion = DiffusionModelPyTorch(args).to(device)
    diffusion.load()
    diffusion.eval()
    for p in diffusion.parameters():
        p.requires_grad = False

    loss_module = GenerationGANLoss(args).to(device)

    if getattr(args, "distributed", False):
        decoder.tokenizer.vae.decoder = DDP(
            decoder.tokenizer.vae.decoder,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        loss_module = DDP(loss_module, device_ids=[args.gpu], find_unused_parameters=False)
    loss_module.train()

    decoder_params = list(unwrap(decoder.tokenizer.vae.decoder).parameters())
    disc_params = list(unwrap(loss_module).discriminator.parameters())
    optimizer_dec = torch.optim.AdamW(
        decoder_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    optimizer_disc = torch.optim.AdamW(
        disc_params, lr=args.disc_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        load_ddp_decoder_state(decoder, checkpoint["decoder"])
        unwrap(loss_module).load_state_dict(checkpoint["loss"])
        optimizer_dec.load_state_dict(checkpoint["optimizer_dec"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_meter = LossManager()
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        for step, (real_images, labels) in enumerate(train_loader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                z_gen = diffusion.generate(
                    batch_size=real_images.size(0),
                    labels=labels,
                    num_steps=args.num_steps,
                    use_cfg=args.use_cfg,
                    device=device,
                )

            fake_images = decoder(z_gen)

            optimizer_dec.zero_grad(set_to_none=True)
            set_requires_grad(unwrap(loss_module).discriminator, False)
            g_loss, g_pack = unwrap(loss_module).generator_loss(fake_images)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            optimizer_dec.step()

            disc_active = epoch >= args.disc_start_epoch
            if disc_active:
                set_requires_grad(unwrap(loss_module).discriminator, True)
                with torch.no_grad():
                    z_gen_d = diffusion.generate(
                        batch_size=real_images.size(0),
                        labels=labels,
                        num_steps=args.num_steps,
                        use_cfg=args.use_cfg,
                        device=device,
                    )
                    fake_images_d = decoder(z_gen_d)
                optimizer_disc.zero_grad(set_to_none=True)
                d_loss, d_pack = loss_module(
                    mode="discriminator",
                    real_images=real_images,
                    fake_images=fake_images_d,
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
                        "decoder": ddp_decoder_state_dict(decoder),
                        "model": tokenizer_prefixed_state(decoder),
                        "loss": unwrap(loss_module).state_dict(),
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
        print("Extension-GAR training complete.")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
