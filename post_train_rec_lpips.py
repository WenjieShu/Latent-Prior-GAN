"""Latent Prior GAN training with a GAR-style reconstruction anchor."""

import os
import sys
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import config
from data.dataloader import build_dataloader
from models.losses import DinoGANLoss
from models.losses_rec_lpips import RecLPIPSLoss
from models.model import LatentPriorGenerator
from utils.distributed import init_distributed_mode
from utils.util import Logger, LossManager, Pack, save_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def reconstruct_with_imf(model, real_images, labels, args):
    """GAR-style reconstruction: E(x) -> noisy latent -> frozen iMF -> decoder."""
    with torch.no_grad():
        z_real = model.tokenizer.encode(real_images)
        z_denoised = model.diffusion.denoise_from_encoder_latent(
            z_real,
            maximum_noise_level=args.reconstruction_noise_level,
            normalized=args.reconstruction_normalized,
            labels=labels,
            num_steps=args.reconstruction_num_steps,
            use_cfg=args.reconstruction_use_cfg,
            fixed_noise=True,
            minimum_noise_level=args.reconstruction_noise_level,
        )
    return model.decode(z_denoised)


def merge_pack(dst, src):
    for key, value in src.items():
        dst[key] = value


def main_worker(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if is_main_process():
        print("=" * 72)
        print("Latent Prior GAN start rec+LPIPS training")
        print("=" * 72)
        print(f"Reconstruction noise level: {args.reconstruction_noise_level}")
        print(f"GAN / rec / LPIPS weights: {args.gan_weight} / {args.rec_weight} / {args.lpips_weight}")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else args.saver_name_pre
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
            print(f"[wandb] Initialized: {args.wandb_project}/{wandb_run_name}")

    model = LatentPriorGenerator(args)
    if is_main_process():
        tokenizer_para = sum(p.numel() for p in model.tokenizer.parameters())
        print(f"Tokenizer Parameters: {tokenizer_para / 1e6:.2f}M")
        if model.diffusion is not None:
            print(f"Diffusion Model: {args.model_type} (frozen)")

    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    gan_losses = DinoGANLoss(args).to(device)
    rec_losses = RecLPIPSLoss(args).to(device)
    rec_losses.eval()

    if is_main_process():
        disc_para_count = sum(p.numel() for p in gan_losses.discriminator.parameters())
        print(f"Discriminator Parameters: {disc_para_count / 1e6:.2f}M")

    if args.train_generator == False:
        model_para = list(model.tokenizer.vae.decoder.parameters())
    else:
        model_para = list(model.tokenizer.vae.decoder.parameters()) + list(model.diffusion.model.parameters())
    disc_para = list(gan_losses.discriminator.parameters())

    optimizer = torch.optim.AdamW(model_para, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer_disc = torch.optim.AdamW(disc_para, lr=args.disc_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    resume_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("discriminator") is not None:
            gan_losses.discriminator.load_state_dict(checkpoint["discriminator"])
        if "optimizer_disc" in checkpoint:
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {resume_epoch}")

    train_loader, train_sampler, train_size = build_dataloader(args, split="train")

    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()
    model.module.tokenizer.vae.encoder.eval()
    if model.module.diffusion is not None:
        if args.train_generator == False:
            model.module.diffusion.eval()
        else:
            model.module.diffusion.train()

    gan_losses = DDP(gan_losses.to(device), device_ids=[args.gpu])
    gan_losses.train()
    rec_losses = rec_losses.to(device)
    rec_losses.eval()

    train_loss = LossManager()
    if is_main_process():
        print("\nStart training...")

    start_epoch = resume_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n{'=' * 60}")
            print(
                f"Epoch {epoch}/{args.epochs}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                f"Disc LR: {optimizer_disc.param_groups[0]['lr']:.6f}"
            )
            print(f"{'=' * 60}")

        start_time = time.time()
        for step, (real_images, labels) in enumerate(train_loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                real_images = real_images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if args.disc_lr_decay_steps > 0 and global_step >= args.warmup_steps:
                    decay_step = min(global_step - args.warmup_steps, args.disc_lr_decay_steps)
                    decay_ratio = decay_step / args.disc_lr_decay_steps
                    current_disc_lr = args.disc_lr + (args.lr - args.disc_lr) * decay_ratio
                    for param_group in optimizer_disc.param_groups:
                        param_group["lr"] = current_disc_lr

                optimizer_disc.zero_grad()
                with torch.no_grad():
                    fake_images = model(labels=labels)
                d_loss, d_loss_pack = gan_losses(
                    inputs=real_images,
                    generations=fake_images,
                    optimizer_idx=1,
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_para, 1.0)
                optimizer_disc.step()

                if global_step >= args.warmup_steps:
                    optimizer.zero_grad()
                    fake_images = model(labels=labels)
                    gan_loss, gan_loss_pack = gan_losses(
                        inputs=real_images,
                        generations=fake_images,
                        optimizer_idx=0,
                    )
                    reconstructions = reconstruct_with_imf(
                        model.module,
                        real_images,
                        labels,
                        args,
                    )
                    rec_lpips_loss, rec_loss_pack = rec_losses(real_images, reconstructions)
                    gen_loss = args.gan_weight * gan_loss + rec_lpips_loss
                    gen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                    optimizer.step()

                    gen_loss_pack = Pack()
                    merge_pack(gen_loss_pack, gan_loss_pack)
                    gen_loss_pack["gan_loss"] = gan_loss
                    gen_loss_pack["gen_loss"] = gen_loss
                    merge_pack(gen_loss_pack, rec_loss_pack)
                    merge_pack(d_loss_pack, gen_loss_pack)

                torch.cuda.synchronize()

            global_step += 1
            d_loss_pack["disc_lr"] = optimizer_disc.param_groups[0]["lr"]
            train_loss.add_loss(d_loss_pack)
            if is_main_process() and step % args.log_interval == 0:
                msg = train_loss.pprint(
                    window=args.log_interval,
                    prefix=f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}]",
                )
                print(msg)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {f"train/{k}": train_loss.get_avg(k, args.log_interval)
                         for k in train_loss.losses},
                        step=global_step,
                    )

        if is_main_process():
            print(f"Epoch {epoch} finished in {time.time() - start_time:.1f}s")
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(
                    args.checkpoint_dir,
                    f"checkpoint-{args.saver_name_pre}-epoch{epoch:03d}.pth.tar",
                )
                save_checkpoint(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "discriminator": gan_losses.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": vars(args),
                    },
                    is_best=False,
                    filename=ckpt_path,
                )

        train_loss.clear()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main_process():
        print("Latent Prior GAN rec+LPIPS training complete.")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    os.environ["NCCL_TIMEOUT_IN_MS"] = "7200000"

    args = config.parse_arg()
    dict_args = vars(args)

    sysout_path = os.path.join(args.results_dir, "logs")
    os.makedirs(sysout_path, exist_ok=True)
    sys.stdout = Logger(sysout_path, args.saver_name_pre)

    if is_main_process():
        print("Configuration:")
        for k, v in dict_args.items():
            print(f"  {k}: {v}")
        print()

    main_worker(args)
