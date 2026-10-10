"""Train LP-GANs across continuous and TiTok latent priors."""

import os
import sys
import time

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from data.dataloader import build_dataloader
from models.model import LatentPriorGenerator
from models.paired_supervision_losses import SupervisedDinoGANLoss
from models.wo_ps_losses import DinoGANLoss
from utils.checkpointing import (
    advance_decoder_lr,
    load_model_state,
    restore_loss_ema,
    training_checkpoint,
    validate_epoch_resume,
    validate_external_prior_identity,
    validate_training_objective,
)
from utils.distributed import finish_distributed, init_distributed_mode, is_main_process
from utils.util import Logger, LossManager, save_checkpoint

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")


def build_optimizers(model, losses, args):
    options = dict(betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, **options)
    optimizer_disc = torch.optim.AdamW(
        losses.discriminator.parameters(), lr=args.disc_lr, **options
    )
    return optimizer, optimizer_disc


def build_continuous_loss(args):
    """Select the audited full objective or LP-GANs w/o PS objective."""
    return DinoGANLoss(args) if args.wo_ps else SupervisedDinoGANLoss(args)


def continuous_forward(model, images, labels, wo_ps):
    return model(None if wo_ps else images, labels=labels)


def continuous_loss(
    losses, real_images, generated_images, paired_images, optimizer_idx, last_layer, wo_ps
):
    if wo_ps:
        return losses(inputs=real_images, generations=generated_images, optimizer_idx=optimizer_idx)
    kwargs = {"last_layer": last_layer} if optimizer_idx == 0 else {}
    return losses(
        inputs=real_images,
        generations=generated_images,
        supervisions=paired_images,
        optimizer_idx=optimizer_idx,
        **kwargs,
    )


def train_continuous(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.use_wandb and not WANDB_AVAILABLE:
        raise ImportError("--use_wandb requires pip install wandb")
    require_cuda()

    init_distributed_mode(args)
    device = args.gpu
    torch.cuda.set_device(device)

    if is_main_process():
        print("=" * 72)
        print("LP-GANs w/o PS" if args.wo_ps else "LP-GANs w/ PS")
        print("=" * 72)
        if not args.wo_ps:
            print(f"Paired supervision noise level: {args.supervision_noise_level}")
            print("Objective: MSE + perceptual + adaptive_weight * disc_weight * hinge GAN")
        else:
            print("Objective: hinge GAN without paired supervision")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else args.saver_name_pre
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
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

    losses = build_continuous_loss(args).to(device)
    if is_main_process():
        disc_para_count = sum(p.numel() for p in losses.discriminator.parameters())
        print(f"Discriminator Parameters: {disc_para_count / 1e6:.2f}M")

    model_para = model.trainable_parameters()
    disc_para = list(losses.discriminator.parameters())

    optimizer, optimizer_disc = build_optimizers(model, losses, args)

    resume_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_epoch_resume(checkpoint)
        validate_training_objective(checkpoint, args)
        restore_loss_ema(losses, checkpoint)
        validate_external_prior_identity(checkpoint, args)
        load_model_state(model, checkpoint["model"], args.prior_family)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("discriminator") is not None:
            losses.discriminator.load_state_dict(checkpoint["discriminator"])
        if "optimizer_disc" in checkpoint:
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if args.reset_resume_lr:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            for group in optimizer_disc.param_groups:
                group["lr"] = args.disc_lr
        resume_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {resume_epoch}")

    train_loader, train_sampler, _ = build_dataloader(args, split="train")
    if len(train_loader) == 0:
        raise ValueError("Training requires at least one full batch per GPU")

    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.module.train_mode()

    losses = DDP(losses.to(device), device_ids=[args.gpu])
    losses.train()
    if not args.wo_ps:
        losses.module.perceptual_loss.eval()

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

                optimizer.zero_grad()
                generated_images, paired_images = continuous_forward(
                    model, real_images, labels, args.wo_ps
                )

                # Decoder update precedes the discriminator update.
                gen_loss, gen_loss_pack = continuous_loss(
                    losses,
                    real_images,
                    generated_images,
                    paired_images,
                    0,
                    model.module.last_layer,
                    args.wo_ps,
                )
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                optimizer.step()

                optimizer_disc.zero_grad()
                d_loss, d_loss_pack = continuous_loss(
                    losses,
                    real_images,
                    generated_images,
                    paired_images,
                    1,
                    model.module.last_layer,
                    args.wo_ps,
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_para, 1.0)
                optimizer_disc.step()

                for key, value in d_loss_pack.items():
                    gen_loss_pack[key] = value
                torch.cuda.synchronize()

            global_step += 1
            if args.wo_ps:
                gen_loss_pack["disc_lr"] = optimizer_disc.param_groups[0]["lr"]
            train_loss.add_loss(gen_loss_pack)
            if is_main_process() and step % args.log_interval == 0:
                msg = train_loss.pprint(
                    window=args.log_interval,
                    prefix=f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}]",
                )
                print(msg)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            f"train/{k}": train_loss.get_avg(k, args.log_interval)
                            for k in train_loss.losses
                        },
                        step=global_step,
                    )

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
                    training_checkpoint(
                        model.module,
                        losses.module,
                        optimizer,
                        optimizer_disc,
                        args,
                        epoch,
                        global_step,
                        step == len(train_loader),
                    ),
                    is_best=False,
                    filename=ckpt_path,
                )

        train_loss.clear()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if args.debug_steps > 0 and global_step >= args.debug_steps:
            break

    if is_main_process():
        print("LP-GANs training complete.")
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def train_titok(args):
    """Preserve the token-pool experiment's update and LR schedule."""
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from data.titok_dataset import PairedARLatentDataset
    from models.titok_losses import ARPairwiseGANLoss
    from models.titok_model import ARLatentPriorDecoder

    require_cuda()
    init_distributed_mode(args, allow_single=True, timeout=7200)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.distributed else "cuda")
    dataset = PairedARLatentDataset(
        args.real_latent_pool, args.generated_latent_pool, args.dataset_dir, args.resolution
    )
    sampler = DistributedSampler(dataset, shuffle=True) if args.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("Training requires nonempty batches")
    if args.debug_steps > len(loader) * args.epochs:
        raise ValueError("Dataset/epochs cannot reach requested debug_steps")
    model = ARLatentPriorDecoder(args.tokenizer_name, args.tokenizer_ckpt_path).to(device)
    losses = ARPairwiseGANLoss(
        args.pairwise_weight,
        args.disc_weight,
        args.lecam_loss_weight,
        args.disc_cr_loss_weight,
    ).to(device)
    model = DDP(model, device_ids=[args.gpu]) if args.distributed else model
    losses = DDP(losses, device_ids=[args.gpu]) if args.distributed else losses
    model_module = model.module if args.distributed else model
    loss_module = losses.module if args.distributed else losses
    model_module.train_mode()
    optimizer, optimizer_disc = build_optimizers(model_module, loss_module, args)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    manager = LossManager()

    global_step = 0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_epoch_resume(checkpoint)
        validate_training_objective(checkpoint, args)
        saved = checkpoint.get("args", {})
        for key in ("tokenizer_name", "lr_decay", "min_lr"):
            if key in saved and saved[key] != getattr(args, key):
                raise ValueError(f"Resume configuration differs for {key}")
        model_module.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_module.discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        restore_loss_ema(loss_module, checkpoint)
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        # Experimental checkpoints were saved before the end-of-epoch LR decay.
        advance_decoder_lr(optimizer, args.lr_decay, args.min_lr)
        if args.debug_steps and global_step >= args.debug_steps:
            raise ValueError("Resume checkpoint already reached debug_steps")
    stopped = False
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, (real_tokens, generated_tokens, real_images) in enumerate(loader, start=1):
            real_tokens = real_tokens.to(device, non_blocking=True)
            generated_tokens = generated_tokens.to(device, non_blocking=True)
            real_images = real_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            generated_images, paired_images = model(real_tokens, generated_tokens)
            g_loss, pack = losses(
                paired_images,
                generated_images,
                real_images,
                optimizer_idx=0,
                last_layer=model_module.last_layer,
            )
            if not torch.isfinite(g_loss):
                raise FloatingPointError(f"Non-finite G loss at step {global_step + 1}")
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_module.trainable_parameters(), 1.0)
            optimizer.step()
            optimizer_disc.zero_grad(set_to_none=True)
            d_loss, d_pack = losses(
                paired_images.detach(), generated_images.detach(), real_images, optimizer_idx=1
            )
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"Non-finite D loss at step {global_step + 1}")
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.discriminator.parameters(), 1.0)
            optimizer_disc.step()
            global_step += 1
            pack.update(d_pack)
            manager.add_loss(pack)
            if global_step == 1 or global_step % args.log_interval == 0:
                if is_main_process():
                    print(
                        f"Step {global_step}: {manager.pprint(window=args.log_interval)}",
                        flush=True,
                    )
            if args.debug_steps > 0 and global_step >= args.debug_steps:
                stopped = True
                break
        if is_main_process():
            tag = f"step{global_step:06d}" if args.debug_steps > 0 else f"epoch{epoch:03d}"
            path = os.path.join(
                args.checkpoint_dir, f"checkpoint-{args.saver_name_pre}-{tag}.pth.tar"
            )
            save_checkpoint(
                training_checkpoint(
                    model_module,
                    loss_module,
                    optimizer,
                    optimizer_disc,
                    args,
                    epoch,
                    global_step,
                    step == len(loader),
                ),
                is_best=False,
                filename=path,
            )
            print(f"Epoch {epoch}: {manager.pprint()}")
        manager.clear()
        advance_decoder_lr(optimizer, args.lr_decay, args.min_lr)
        next_lr = optimizer.param_groups[0]["lr"]
        if is_main_process():
            current_disc_lr = optimizer_disc.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} complete: decoder_lr={next_lr:.8g}, disc_lr={current_disc_lr:.8g}",
                flush=True,
            )
        if args.distributed:
            dist.barrier()
        if stopped:
            break


def main(argv=None):
    os.environ["NCCL_TIMEOUT_IN_MS"] = "7200000"

    args = config.parse_arg(argv)
    sysout_path = os.path.join(args.results_dir, "logs")
    os.makedirs(sysout_path, exist_ok=True)
    logger = Logger(sysout_path, args.saver_name_pre) if is_main_process() else None
    previous_stdout = sys.stdout
    try:
        if logger is not None:
            sys.stdout = logger
            print("Configuration:")
            for key, value in vars(args).items():
                print(f"  {key}: {value}")
            print()
        if args.prior_family == "titok":
            train_titok(args)
        else:
            train_continuous(args)
    finally:
        sys.stdout = previous_stdout
        if logger is not None:
            logger.close()
        finish_distributed()


if __name__ == "__main__":
    main()
