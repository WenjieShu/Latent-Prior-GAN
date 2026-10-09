"""Latent Prior GAN training entry point."""
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
from models.model import LatentPriorGenerator
from models.pure_gan_losses import DinoGANLoss
from utils.distributed import init_distributed_mode
from utils.checkpointing import (
    loss_ema_state,
    restore_loss_ema,
    validate_epoch_resume,
    external_prior_metadata,
    load_model_state,
    model_state_for_checkpoint,
    validate_external_prior_identity,
)
from utils.util import Logger, LossManager, save_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def is_main_process():
    """Check if current process is main (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get('RANK', 0)) == 0

def main_worker(args):
    """Main training worker."""
    if args.use_wandb and not WANDB_AVAILABLE:
        raise ImportError("--use_wandb requires pip install wandb")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    rank = dist.get_rank()
    device = args.gpu
    torch.cuda.set_device(device)

    if is_main_process():
        print("=" * 72)
        print("Latent Prior GAN start training")
        print("=" * 72)
        # Initialize wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else args.saver_name_pre
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config=vars(args),
            )
            print(f"[wandb] Initialized: {args.wandb_project}/{wandb_run_name}")

    model = LatentPriorGenerator(args)
    if is_main_process():
        tokenizer_para = sum(p.numel() for p in model.tokenizer.parameters())
        print(f"Tokenizer Parameters: {tokenizer_para / 1e6:.2f}M")
        
        if model.diffusion is not None:
            name = args.model_type if args.prior_family == "imf" else args.prior_family
            print(f"Latent prior: {name} (frozen)")
        
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    losses = DinoGANLoss(args).to(device)
    if is_main_process():
        if losses.discriminator is not None:
            disc_para_count = sum(p.numel() for p in losses.discriminator.parameters())
            print(f"Discriminator Parameters: {disc_para_count / 1e6:.2f}M")
        else:
            print("Discriminator disabled")

    model_para = model.tokenizer.get_decoder_params()
    disc_para = list(losses.discriminator.parameters()) 

    optimizer = torch.optim.AdamW(model_para, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer_disc = torch.optim.AdamW(disc_para, lr=args.disc_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    resume_epoch = 0
    global_step = 0
    ### copied from GAR-FID, needs slight revision
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_epoch_resume(checkpoint)
        restore_loss_ema(losses, checkpoint)
        validate_external_prior_identity(checkpoint, args)
        load_model_state(model, checkpoint['model'], args.prior_family)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if losses.discriminator is not None and checkpoint.get('discriminator') is not None:
            losses.discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_disc is not None and 'optimizer_disc' in checkpoint:
            optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
        if args.reset_resume_lr:
            for group in optimizer.param_groups:
                group['lr'] = args.lr
            for group in optimizer_disc.param_groups:
                group['lr'] = args.disc_lr
        resume_epoch = int(checkpoint.get('epoch', 0))
        global_step = int(checkpoint.get('global_step', 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {resume_epoch}")
            
    train_loader, train_sampler, train_size = build_dataloader(args, split="train")
    if len(train_loader) == 0:
        raise ValueError("Training requires at least one full batch per GPU")

    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()
    model.module.tokenizer.encoder_eval()
    model.module.diffusion.eval()

    losses = DDP(losses.to(device), device_ids=[args.gpu])
    losses.train()
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

            with torch.autocast(device_type='cuda', dtype=torch.float32):
                real_images = real_images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                fake_images = model(labels=labels)
                gen_loss, gen_loss_pack = losses(
                    inputs = real_images,
                    generations = fake_images,
                    optimizer_idx=0,
                    )
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                optimizer.step()

                optimizer_disc.zero_grad()
                d_loss, d_loss_pack = losses(
                        inputs = real_images,
                        generations = fake_images,
                    optimizer_idx=1,
                        )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_para, 1.0)
                optimizer_disc.step()

                torch.cuda.synchronize()
                for key, value in gen_loss_pack.items():
                    d_loss_pack[key] = value

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
                    wandb.log({f"train/{k}": train_loss.get_avg(k, args.log_interval)
                               for k in train_loss.losses}, step=global_step)

            if args.debug_steps > 0 and global_step >= args.debug_steps:
                break

        if is_main_process():
            print(f"Epoch {epoch} finished in {time.time() - start_time:.1f}s")
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(
                    args.checkpoint_dir,
                    f"checkpoint-{args.saver_name_pre}-epoch{epoch:03d}.pth.tar",
                )
                save_checkpoint(
                    {
                        "model": model_state_for_checkpoint(model.module, args.prior_family),
                        "optimizer": optimizer.state_dict(),
                        "discriminator": (
                            losses.module.discriminator.state_dict()
                            if losses.module.discriminator is not None else None
                        ),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "epoch": epoch,
                        "epoch_complete": step == len(train_loader),
                        "loss_ema": loss_ema_state(losses.module),
                        "global_step": global_step,
                        "args": vars(args),
                        **external_prior_metadata(args),
                    },
                    is_best=False,
                    filename=ckpt_path,
                )

        train_loss.clear()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if args.debug_steps > 0 and global_step >= args.debug_steps:
            break

    if is_main_process():
        print("Latent Prior GAN training complete.")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == '__main__':
    os.environ['NCCL_TIMEOUT_IN_MS'] = '7200000'
    
    args = config.parse_arg(objective="pure")
    dict_args = vars(args)
    
    # Setup logging
    sysout_path = os.path.join(args.results_dir, "logs")
    os.makedirs(sysout_path, exist_ok=True)
    if is_main_process():
        sys.stdout = Logger(sysout_path, args.saver_name_pre)
    
    if is_main_process():
        print("Configuration:")
        for k, v in dict_args.items():
            print(f"  {k}: {v}")
        print()
    
    main_worker(args)
