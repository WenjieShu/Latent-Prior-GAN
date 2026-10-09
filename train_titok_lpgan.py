"""Train a TiTok decoder from paired precomputed real/AR token pools."""

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.titok_dataset import PairedARLatentDataset
from models.titok_model import ARLatentPriorDecoder
from models.titok_losses import ARPairwiseGANLoss
from utils.titok_distributed import init_distributed_mode
from utils.util import LossManager, save_checkpoint
from utils.checkpointing import advance_decoder_lr, loss_ema_state, restore_loss_ema, validate_epoch_resume


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_latent_pool", required=True)
    parser.add_argument("--generated_latent_pool", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tokenizer_name", required=True, choices=["titok-vql32", "titok-vqb64"])
    parser.add_argument("--tokenizer_ckpt_path", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr", type=float, default=1e-5)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--pairwise_weight", type=float, default=1.0)
    parser.add_argument("--disc_weight", type=float, default=1.0)
    parser.add_argument("--lecam_loss_weight", type=float, default=0.001)
    parser.add_argument("--disc_cr_loss_weight", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--resume", default="", help="completed-epoch LP-GAN checkpoint")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="stop after this many optimizer steps; 0 runs all epochs")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    import math
    if min(args.batch_size, args.epochs) <= 0 or args.workers < 0 or args.max_steps < 0:
        parser.error("Invalid batch size, epoch count, workers, or max_steps")
    if args.resolution != 256 or not math.isfinite(args.lr_decay) or not 0 < args.lr_decay <= 1:
        parser.error("TiTok uses resolution 256 and lr_decay in (0,1]")
    for key in ("lr", "disc_lr", "min_lr", "pairwise_weight", "disc_weight", "lecam_loss_weight", "disc_cr_loss_weight", "weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"Invalid --{key}")
    return args


def main():
    args = parse_args()
    init_distributed_mode(args)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.distributed else "cuda")
    dataset = PairedARLatentDataset(args.real_latent_pool, args.generated_latent_pool,
                                    args.dataset_dir, args.resolution)
    sampler = DistributedSampler(dataset, shuffle=True) if args.distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0)
    if len(loader) == 0 or args.max_steps < 0:
        raise ValueError("Training requires nonempty batches and nonnegative max_steps")
    if args.max_steps > len(loader) * args.epochs:
        raise ValueError("Dataset/epochs cannot reach requested max_steps")
    model = ARLatentPriorDecoder(args.tokenizer_name, args.tokenizer_ckpt_path).to(device)
    losses = ARPairwiseGANLoss(
        args.pairwise_weight, args.disc_weight,
        args.lecam_loss_weight, args.disc_cr_loss_weight,
    ).to(device)
    model = DDP(model, device_ids=[args.gpu]) if args.distributed else model
    losses = DDP(losses, device_ids=[args.gpu]) if args.distributed else losses
    model_module = model.module if args.distributed else model
    loss_module = losses.module if args.distributed else losses
    optimizer = torch.optim.AdamW(
        model_module.trainable_parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    optimizer_disc = torch.optim.AdamW(
        loss_module.discriminator.parameters(), lr=args.disc_lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    manager = LossManager()

    global_step = 0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_epoch_resume(checkpoint)
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
        if args.max_steps and global_step >= args.max_steps:
            raise ValueError("Resume checkpoint already reached max_steps")
    stopped = False
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, (real_tokens, generated_tokens, target_images) in enumerate(loader, start=1):
            real_tokens = real_tokens.to(device, non_blocking=True)
            generated_tokens = generated_tokens.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            real_images, fake_images = model(real_tokens, generated_tokens)
            g_loss, pack = losses(
                real_images, fake_images, target_images,
                optimizer_idx=0, last_layer=model_module.last_layer,
            )
            if not torch.isfinite(g_loss):
                raise FloatingPointError(f"Non-finite G loss at step {global_step + 1}")
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_module.trainable_parameters(), 1.0)
            optimizer.step()
            optimizer_disc.zero_grad(set_to_none=True)
            d_loss, d_pack = losses(real_images.detach(), fake_images.detach(),
                                    target_images, optimizer_idx=1)
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"Non-finite D loss at step {global_step + 1}")
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.discriminator.parameters(), 1.0)
            optimizer_disc.step()
            global_step += 1
            pack.update(d_pack)
            manager.add_loss(pack)
            if global_step == 1 or global_step % 50 == 0:
                if not args.distributed or dist.get_rank() == 0:
                    print(f"Step {global_step}: {manager.pprint(window=50)}", flush=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                stopped = True
                break
        if not args.distributed or dist.get_rank() == 0:
            tag = f"step{global_step:06d}" if args.max_steps > 0 else f"epoch{epoch:03d}"
            path = os.path.join(args.checkpoint_dir, f"checkpoint-titok-lpg-{tag}.pth.tar")
            save_checkpoint({"model": model_module.state_dict(), "optimizer": optimizer.state_dict(),
                             "discriminator": loss_module.discriminator.state_dict(),
                             "optimizer_disc": optimizer_disc.state_dict(), "epoch": epoch,
                             "global_step": global_step, "args": vars(args),
                             "epoch_complete": step == len(loader),
                             "loss_ema": loss_ema_state(loss_module)}, is_best=False, filename=path)
            print(f"Epoch {epoch}: {manager.pprint()}")
        manager.clear()
        advance_decoder_lr(optimizer, args.lr_decay, args.min_lr)
        next_lr = optimizer.param_groups[0]["lr"]
        if not args.distributed or dist.get_rank() == 0:
            current_disc_lr = optimizer_disc.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} complete: decoder_lr={next_lr:.8g}, "
                f"disc_lr={current_disc_lr:.8g}",
                flush=True,
            )
        if args.distributed:
            dist.barrier()
        if stopped:
            break
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
