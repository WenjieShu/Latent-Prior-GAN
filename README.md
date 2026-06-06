# iMF Decoder Post-Training

This repository contains clean training code for two decoder-side iMF/GAR
post-training paths:

- **GAR**: adapt the SD-VAE decoder on denoised encoder latents with
  reconstruction, perceptual, Gram, and adversarial losses.
- **Extension-GAR**: adapt the SD-VAE decoder on pure generated iMF latents with
  a GAN-only objective.

## Objectives

GAR uses paired reconstruction:

```text
x -> frozen SD-VAE encoder -> z_e
z_e -> add noise eta_t -> z_t
z_t, y -> frozen iMF denoiser -> z_d
z_d -> trainable SD-VAE decoder -> x_rec
x_rec vs. x -> MSE + perceptual/LPIPS + Gram + GAN
```

Extension-GAR is generation-side and unpaired:

```text
epsilon, y -> frozen iMF generator -> z_g
z_g -> trainable SD-VAE decoder -> x_fake
real ImageNet images vs. x_fake -> GAN loss
```

Extension-GAR does **not** use:

- paired reconstruction targets;
- VAE encoder latents `z_e`;
- `denoise_from_encoder_latent`;
- LPIPS/perceptual loss;
- Gram loss;
- reconstruction MSE.

Both paths train only `tokenizer.vae.decoder`; the encoder, iMF generator, and
the rest of the VAE remain frozen.

## Main Files

```text
post_train.py
    Backward-compatible extension-GAR training entry point.

post_train_extension_gar.py
    Explicit extension-GAR training entry point.

post_train_gar.py
    Clean GAR training entry point.

models/generation_gan_loss.py
    GAN-only objective for extension-GAR.

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

metric/
    Not included in this clean training package; evaluation scripts should live
    in a separate eval package or be added as a clearly separated module.

yaml/gar/
    Clean GAR configs.

yaml/extension_gar/
    Clean extension-GAR configs.

scripts/generated/sbatch_extension_gar_xl_nt1_gan_only.sh
    Slurm template for extension-GAR.

scripts/generated/sbatch_gar_b_fixednoise0p5_gan.sh
    Slurm template for GAR.
```

## Training

Extension-GAR, iMF-XL, nt=1.0, no CFG, GAN-only:

```bash
cd /mnt/nfs/wenjie/IMF-GAP_V2
bash scripts/generated/sbatch_extension_gar_xl_nt1_gan_only.sh
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
  --config yaml/extension_gar/imf_xl_nt1_gan_only.yaml
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
    "decoder": ...,        # GeneratedLatentDecoder state dict
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

- Keep GAR and extension-GAR in separate entry points and loss modules.
- Use `models/gar_loss.py` for reconstruction/perceptual/Gram GAR losses.
- Use `models/generation_gan_loss.py` for extension-GAR GAN-only training.
- Do not commit generated images, NPZs, checkpoints, logs, or metric caches.
