# Latent Prior GAN

This repository contains clean training code for decoder-side iMF experiments:

- **Latent Prior GAN**: adapt the SD-VAE decoder on pure iMF-generated latents
  with an adversarial image-space objective.
- **GAR**: adapt the SD-VAE decoder on denoised encoder latents with
  reconstruction, perceptual, Gram, and adversarial losses.

## Objectives

Latent Prior GAN is generation-side and unpaired:

```text
epsilon, y -> frozen iMF latent prior -> z_g
z_g -> trainable SD-VAE decoder -> x_fake
real ImageNet images vs. x_fake -> adversarial loss
```

It does **not** use paired reconstruction targets, VAE encoder latents,
LPIPS/perceptual loss, Gram loss, or reconstruction MSE.

GAR is kept as a baseline and uses paired reconstruction:

```text
x -> frozen SD-VAE encoder -> z_e
z_e -> add noise eta_t -> z_t
z_t, y -> frozen iMF denoiser -> z_d
z_d -> trainable SD-VAE decoder -> x_rec
x_rec vs. x -> MSE + perceptual/LPIPS + Gram + GAN
```

Both paths train only `tokenizer.vae.decoder`; the encoder, iMF generator, and
the rest of the VAE remain frozen.

## Main Files

```text
post_train.py
    Latent Prior GAN training entry point.

post_train_gar.py
    Clean GAR training entry point.

models/latent_prior_gan_model.py
    Frozen iMF prior, trainable SD-VAE decoder, and DINO GAN objective.

models/dino_gan_loss.py
    GAN loss with a DINO-based discriminator.

models/gar_model.py
    GAR model wrapper: encoder latent -> iMF denoise -> decoder.

models/gar_loss.py
    GAR loss: MSE + perceptual/LPIPS + Gram + adversarial objective.

models/tokenizer.py
    SD-VAE wrapper. The encoder is frozen; only vae.decoder is trainable.

models/diffusion_pytorch.py
    PyTorch iMF wrapper. The generator is frozen and samples z_g from noise.

data/dataloader.py
    ImageNet dataloader for real discriminator samples.

utils/config.py
    YAML/JSON config loading and saving helpers.

utils/checkpointing.py
    Checkpoint state-dict cleanup and resume helpers.

utils/training.py
    Seed, TF32, trainable-parameter, and DDP unwrap helpers.

yaml/gar/
    Clean GAR configs.

yaml/latent_prior_gan/
    Latent Prior GAN configs.

scripts/generated/sbatch_latent_prior_gan_xl_nt1_gan_only.sh
    Slurm template for Latent Prior GAN.

scripts/generated/sbatch_gar_b_fixednoise0p5_gan.sh
    Slurm template for GAR.
```

## Training

Latent Prior GAN, iMF-XL, nt=1.0, no CFG:

```bash
cd /mnt/nfs/wenjie/IMF-GAP_V2
bash scripts/generated/sbatch_latent_prior_gan_xl_nt1_gan_only.sh
```

Direct command:

```bash
export SD_VAE_PATH=/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse
export DINO_CKPT_PATH=/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth
export PYTHONPATH=/mnt/nfs/wenjie/IMF-GAP_V2:$PYTHONPATH

/mnt/nfs/wenjie/FD-Loss/.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  post_train.py \
  --config yaml/latent_prior_gan/imf_xl_nt1_gan_only.yaml
```

GAR, iMF-B, fixed noise eta_t=0.5:

```bash
cd /mnt/nfs/wenjie/IMF-GAP_V2
bash scripts/generated/sbatch_gar_b_fixednoise0p5_gan.sh
```

Direct command:

```bash
/mnt/nfs/wenjie/FD-Loss/.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  post_train_gar.py \
  --config yaml/gar/imf_b_fixednoise0p5_gan.yaml
```

## Checkpoints

V2 checkpoints are saved as:

```python
{
    "decoder": ...,        # tokenizer state dict with trained vae.decoder
    "model": ...,          # tokenizer-prefixed state dict for eval compatibility
    "loss": ...,           # GAN loss/discriminator state
    "optimizer_dec": ...,
    "optimizer_disc": ...,
    "epoch": ...,
    "global_step": ...,
    "args": ...,
}
```

The `model` key is intentionally compatible with existing GAR eval scripts that
load post-trained decoder weights from `tokenizer.*` keys.

During distributed training, only `tokenizer.vae.decoder` is wrapped with DDP
and optimized. The rest of the VAE, including the encoder and post-quant
projection, stays frozen.

## External Artifacts

Large files are not part of the clean code checkout. Expected paths on the
cluster are:

```text
/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-XL-2.pt
/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse/
/mnt/nfs/wenjie/dataset/imagenet_idx/
/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth
```

Outputs should stay under:

```text
/mnt/nfs/wenjie/IMF-GAP_V2/checkpoints/
/mnt/nfs/wenjie/IMF-GAP_V2/results/
/mnt/nfs/wenjie/IMF-GAP_V2/logs/
```

## Code Hygiene Rules

- Keep Latent Prior GAN and GAR in separate entry points and loss modules.
- Use `models/gar_loss.py` for reconstruction/perceptual/Gram GAR losses.
- Use `models/dino_gan_loss.py` for the DINO-discriminator GAN objective.
- Do not commit generated images, NPZs, checkpoints, logs, or metric caches.
