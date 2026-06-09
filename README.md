# Latent Prior GAN

This repository contains clean training code for decoder-side iMF experiments.

- **Latent Prior GAN**: adapt the SD-VAE decoder on pure iMF-generated latents
  with an adversarial image-space objective.

## Objectives

Latent Prior GAN is generation-side and unpaired:

```text
epsilon, y -> frozen iMF latent prior -> z_g
z_g -> trainable SD-VAE decoder -> x_fake
real ImageNet images vs. x_fake -> adversarial loss
```

It does **not** use paired reconstruction targets, VAE encoder latents,
LPIPS/perceptual loss, Gram loss, or reconstruction MSE.

The default path trains only `tokenizer.vae.decoder`; the encoder, iMF
generator, and the rest of the VAE remain frozen. Generator-side training is a
planned isolated extension and should use an explicit training mode.

## Main Files

```text
post_train.py
    Latent Prior GAN training entry point.

models/model.py
    Frozen iMF prior and trainable SD-VAE decoder.

models/losses.py
    GAN loss with a DINO-based discriminator.

models/tokenizer.py
    SD-VAE wrapper. The encoder is frozen; only vae.decoder is trainable.

models/diffusion_pytorch.py
    PyTorch iMF wrapper. The generator is frozen and samples z_g from noise.

data/dataloader.py
    ImageNet dataloader for real discriminator samples.

config.py
    Argument parsing, YAML loading, output directory setup, and seed setup.

yaml/latent_prior_gan/
    Latent Prior GAN configs.

scripts/generated/sbatch_latent_prior_gan_xl_nt1_gan_only.sh
    Slurm template for Latent Prior GAN.

scripts/generated/sbatch_lpg_b_eval_openai.sh
    OpenAI gFID w/o CFG evaluation template for the B-model checkpoint.
```

## Training

Latent Prior GAN, iMF-XL, nt=1.0, no CFG:

```bash
cd /mnt/nfs/wenjie/Latent-Prior-GAN
bash scripts/generated/sbatch_latent_prior_gan_xl_nt1_gan_only.sh
```

Direct command:

```bash
export SD_VAE_PATH=/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse
export DINO_CKPT_PATH=/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth
export PYTHONPATH=/mnt/nfs/wenjie/Latent-Prior-GAN:$PYTHONPATH

/mnt/nfs/wenjie/FD-Loss/.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  post_train.py \
  --config yaml/latent_prior_gan/imf_xl_nt1_gan_only.yaml
```

## Checkpoints

Latent Prior GAN checkpoints follow the GAR-FID training format:

```python
{
    "model": ...,          # DDP-unwrapped LatentPriorGenerator state dict
    "optimizer": ...,
    "discriminator": ...,
    "optimizer_disc": ...,
    "epoch": ...,
    "global_step": ...,
    "args": ...,
}
```

The `model` key contains `tokenizer.*` weights, so existing GAR eval scripts can
extract the post-trained decoder weights from it.

During distributed training, the top-level Latent Prior GAN module is wrapped
with DDP. The encoder and frozen iMF prior stay frozen; the decoder is optimized.

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
/mnt/nfs/wenjie/Latent-Prior-GAN/checkpoints/
/mnt/nfs/wenjie/Latent-Prior-GAN/results/
/mnt/nfs/wenjie/Latent-Prior-GAN/logs/
```

## Code Hygiene Rules

- Follow the validated VQ-Transplant/GAR training template for GAN training.
- Do not rewrite `main_worker` or `models/losses.py` with generated code.
- Use `models/losses.py` only for the DINO-discriminator GAN objective.
- Keep model/generation code in `models/model.py` and loss code in
  `models/losses.py`.
- Do not add duplicate directories or duplicate modules for analysis
  experiments; extend the existing modules instead.
- Do not commit generated images, NPZs, checkpoints, logs, or metric caches.
