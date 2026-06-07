"""
Clean GAR decoder post-training.

This entry point adapts only the SD-VAE decoder on model-denoised encoder
latents, matching the original GAR training objective:

    x -> E_base(x) -> z_e
    z_t = add_noise(z_e, eta_t)
    z_d = G_iMF.denoise(z_t, y)
    x_rec = D_trainable(z_d)

Loss: reconstruction MSE + perceptual/LPIPS + Gram + adversarial loss.
"""

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
from models.gar_loss import PostTrainingLoss
from models.gar_model import TokenizerFlowComposition
from utils.checkpointing import gar_model_state, load_gar_state
from utils.config import load_config, save_config
from utils.distributed import init_distributed_mode, is_main_process
from utils.training import enable_tf32, seed_everything, set_trainable, unwrap
from utils.util import Logger, LossManager, save_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

enable_tf32()


def parse_args():
    parser = argparse.ArgumentParser(description="GAR decoder post-training on denoised encoder latents.")
    parser.add_argument("--config", default="", type=str, help="optional YAML config")

    parser.add_argument("--dataset_dir", default="/mnt/nfs/wenjie/dataset/imagenet_idx")
    parser.add_argument("--dataset_name", default="ImageNet", choices=["ImageNet"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--model_type", default="iMF-B-2", choices=["iMF-B-2", "iMF-M-2", "iMF-L-2", "iMF-XL-2"])
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--vae_type", default="mse", choices=["mse", "ema"])
    parser.add_argument("--pretrained_imf_pytorch", default="")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--maximum_noise_level", type=float, default=0.4)
    parser.add_argument("--fixed_noise", action="store_true", default=False)
    parser.add_argument("--normalized", action="store_true", default=False)
    parser.add_argument("--debug_mode", action="store_true", default=False)
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
    parser.add_argument("--gram_loss_weight", type=float, default=10.0)

    parser.add_argument("--checkpoint_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/checkpoints/gar")
    parser.add_argument("--results_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/results/gar")
    parser.add_argument("--yaml_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/yaml/gar")
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

    if not args.pretrained_imf_pytorch and not args.debug_mode:
        args.pretrained_imf_pytorch = (
            f"/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/{args.model_type}.pt"
        )
    if not args.saver_name_pre:
        noise_tag = str(args.maximum_noise_level).replace(".", "p")
        fixed_tag = "fixed" if args.fixed_noise else "rand"
        args.saver_name_pre = f"gar_{args.model_type}_noise{noise_tag}_{fixed_tag}"
    return args


def fill_cfg_defaults(args):
    cfg = DiffusionModelPyTorch.CFG_PARAMS[args.model_type]
    if args.cfg_omega is None:
        args.cfg_omega = cfg["omega"]
    if args.cfg_t_min is None:
        args.cfg_t_min = cfg["t_min"]
    if args.cfg_t_max is None:
        args.cfg_t_max = cfg["t_max"]
    args.omega = args.cfg_omega
    args.t_min = args.cfg_t_min
    args.t_max = args.cfg_t_max


def main():
    args = parse_args()
    fill_cfg_defaults(args)
    seed_everything(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    init_distributed_mode(args)
    if not torch.cuda.is_available():
        raise RuntimeError("GAR training requires CUDA.")
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
        print("GAR decoder post-training")
        print("=" * 72)
        print("Objective: x -> E(x) -> noise -> frozen iMF denoise -> trainable decoder.")
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

    model = TokenizerFlowComposition(args).to(device)
    set_trainable(model.tokenizer.vae, False)
    set_trainable(model.tokenizer.vae.decoder, True)
    model.tokenizer.vae.encoder.eval()
    model.tokenizer.vae.decoder.train()
    if model.diffusion_pytorch is not None:
        model.diffusion_pytorch.eval()
        set_trainable(model.diffusion_pytorch, False)

    loss_module = PostTrainingLoss(args).to(device)
    if getattr(args, "distributed", False):
        model.tokenizer.vae.decoder = DDP(
            model.tokenizer.vae.decoder,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        loss_module = DDP(loss_module, device_ids=[args.gpu], find_unused_parameters=False)
    loss_module.train()
    unwrap(loss_module).perceptual_loss.eval()
    unwrap(loss_module).gram_loss.eval()

    decoder_params = list(unwrap(model.tokenizer.vae.decoder).parameters())
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
        load_gar_state(model, checkpoint["model"])
        optimizer_dec.load_state_dict(checkpoint["optimizer_dec"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        unwrap(loss_module).discriminator.load_state_dict(checkpoint["discriminator"])
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

            fake_images = model(real_images, labels=labels)

            optimizer_dec.zero_grad(set_to_none=True)
            set_trainable(unwrap(loss_module).discriminator, False)
            g_loss, g_pack = unwrap(loss_module)(
                real_images,
                fake_images,
                optimizer_idx=0,
                cur_epoch=epoch,
                last_layer=model.tokenizer.last_layer,
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            optimizer_dec.step()

            disc_active = epoch >= args.disc_start_epoch
            if disc_active:
                set_trainable(unwrap(loss_module).discriminator, True)
                optimizer_disc.zero_grad(set_to_none=True)
                d_loss, d_pack = loss_module(
                    real_images,
                    fake_images.detach(),
                    optimizer_idx=1,
                    cur_epoch=epoch,
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
                        "model": gar_model_state(model),
                        "discriminator": unwrap(loss_module).discriminator.state_dict(),
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
        print("GAR training complete.")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
