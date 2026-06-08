"""Latent Prior GAN training entry point."""
import os
import sys
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import config
from data.dataloader import build_dataloader
from models.model import LatentPriorGenerator
from models.losses import DinoGANLoss
from utils.checkpointing import load_tokenizer_state, tokenizer_prefixed_state, tokenizer_state
from utils.distributed import init_distributed_mode
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
    return int(os.environ.get('LOCAL_RANK', 0)) == 0

def main_worker(args):
    """Main training worker."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
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
            print(f"Diffusion Model: {args.model_type} (frozen)")
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    losses = DinoGANLoss(args).to(device)
    if is_main_process():
        if losses.discriminator is not None:
            disc_para_count = sum(p.numel() for p in losses.discriminator.parameters())
            print(f"Discriminator Parameters: {disc_para_count / 1e6:.2f}M")
        else:
            print("Discriminator disabled")


    model_para = list(model.tokenizer.vae.decoder.parameters())
    disc_para = list(losses.discriminator.parameters()) if losses.discriminator is not None else []



    train_loader, train_sampler, train_size = build_dataloader(args, split="train")


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

            base_model.set_discriminator_requires_grad(False)
            optimizer_dec.zero_grad(set_to_none=True)
            optimizer_disc.zero_grad(set_to_none=True)
            g_loss, g_pack = model(mode="generator", labels=labels)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            optimizer_dec.step()
            base_model.set_discriminator_requires_grad(True)

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


if __name__ == '__main__':
    os.environ['NCCL_TIMEOUT_IN_MS'] = '7200000'
    
    args = config.parse_arg()
    dict_args = vars(args)
    
    # Setup logging
    sysout_path = os.path.join(args.results_dir, "logs")
    os.makedirs(sysout_path, exist_ok=True)
    sys.stdout = Logger(sysout_path, args.saver_name_pre)
    
    if is_main_process():
        print("Configuration:")
        for k, v in dict_args.items():
            print(f"  {k}: {v}")
        print()
    
    main_worker(args)