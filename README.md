# Latent Prior GAN

LP-GAN adapts the decoder of a latent generative model while freezing the latent prior and image encoder. A discriminator learns from generated images, and paired supervision anchors the decoder to source images. Training has no additional generation-time network.

[Wenjie Shu](https://wenjieshu.github.io/) · [Repository](https://github.com/WenjieShu/Latent-Prior-GAN)

## Supported models

| Frozen prior | Decoder | Paired supervision | Sampling |
|:--|:--|:--|:--|
| iMF B / M / L / XL | SD-VAE MSE | generation-aware reconstruction at noise level η | official `.pth`, registered CFG interval, 1 step |
| DMF XL | SD-VAE MSE | native denoising of perturbed encoder latents | native checkpoint and source checkout, 1 step |
| MeanFlow XL | SD-VAE EMA | native trajectory from perturbed encoder latents | AlphaFlow snapshot loader and EMA VAE, 1 step |
| TiTok L32 / B64 | TiTok decoder modules | cached real tokens paired with center-cropped source images | frozen MaskGIT, 8 steps |

The continuous objective uses MSE and the experimental LPIPS/ConvNeXt perceptual loss, plus adaptively weighted adversarial loss. TiTok uses paired MSE and adaptively weighted adversarial loss. Both preserve the experimental DINO discriminator, LeCAM, consistency regularization, and generator-then-discriminator update order. `train_pure_gan.py` provides the adversarial-only ablation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training and full generation require CUDA. Place ImageNet in `data/imagenet/train/<class>/`. Supply original model weights separately; this repository contains no datasets or pretrained weights. The SD-VAE is loaded from `--vae_path`, the documented environment overrides, or Hugging Face. DINO, VGG, ConvNeXt, and LPIPS projection weights are also needed for continuous training. DINO/VGG/ConvNeXt can download their feature weights; supply the LPIPS projection weights through `LPIPS_VGG_CKPT` (see [setup details](docs/reproduction.md)).

For iMF, put the **official PyTorch** weights in `checkpoints/imf/iMF-{B,M,L,XL}-2.pth`, or pass `--pretrained_imf_pytorch`. Converted iMF `.pt` files are unsupported. DMF/MeanFlow retain their native checkpoint formats. TiTok defaults to `checkpoints/titok/tokenizer_titok_l32.bin` and `generator_titok_l32.bin` (corresponding `b64` files for B64).

## Train

```bash
# LP-GAN: generation and paired CFG supervision.
NPROC_PER_NODE=8 bash scripts/train_lpgan.sh --config configs/imf_b.yaml \
  --dataset_dir data/imagenet --checkpoint_dir checkpoints/runs/imf_b

# Adversarial-only ablation with the same frozen prior.
NPROC_PER_NODE=8 bash scripts/train_pure_gan.sh --config configs/imf_b.yaml \
  --dataset_dir data/imagenet --checkpoint_dir checkpoints/runs/imf_b_pure
```

Batch size is per GPU. The launchers use `torchrun`, including for a single GPU. Choose `configs/imf_m.yaml`, `imf_l.yaml`, or `imf_xl.yaml` for other scales. Explicit CLI values override YAML. Unknown or ignored historical fields fail validation. Generation and paired CFG can be controlled independently with `--no-use_cfg` and `--no-supervision_use_cfg`.

DMF and MeanFlow additionally require their original source checkouts and weights; see [reproduction instructions](docs/reproduction.md). Resume a completed epoch using `--resume`. Restored optimizer learning rates take precedence; `--reset_resume_lr` explicitly applies new continuous-model learning rates. TiTok restores its decoder LR and advances the original epoch decay exactly once.

## TiTok token pools

```bash
# The same source indices and class labels are preserved in both pools.
python precompute_titok_real_tokens.py --dataset_dir data/imagenet \
  --tokenizer_name titok-vql32 --out pools/titok_real.npz
python precompute_titok_generated_tokens.py --generator_name titok-L32 \
  --num_samples 1281167 --label_pool pools/titok_real.npz \
  --deterministic --seed 3407 --out pools/titok_generated.npz
NPROC_PER_NODE=8 bash scripts/train_titok.sh \
  --real_latent_pool pools/titok_real.npz \
  --generated_latent_pool pools/titok_generated.npz \
  --dataset_dir data/imagenet --checkpoint_dir checkpoints/runs/titok_l32
```

Set `--num_samples` to the actual number of rows in the real pool. Generated pools carry `image_indices` as well as labels; training verifies source paths, class labels, center-crop metadata, and tokenizer/generator pairing. Streamed generation can resume with `--deterministic`; incompatible shards require `--restart`. `check_titok_pairing.py` provides a diagnostic re-encoding and image grid.

## Generate and evaluate

```bash
# Omit --decoder_checkpoint to evaluate the original decoder.
NPROC_PER_NODE=8 bash scripts/eval_generation.sh --config configs/imf_b.yaml \
  --decoder_checkpoint checkpoints/runs/imf_b/checkpoint-LP_GAN.pth.tar \
  --num_samples 50000 --out_dir results/imf_b_adapted

# TiTok uses its registered eight sampling steps.
NPROC_PER_NODE=8 bash scripts/eval_generation.sh --prior_family titok \
  --generator_name titok-L32 --decoder_checkpoint checkpoints/runs/titok_l32/checkpoint-titok-lpg-epoch002.pth.tar \
  --out_dir results/titok_l32_adapted

pip install -r requirements-eval.txt
python eval_fid.py --protocol torch_fidelity \
  --samples results/imf_b_adapted/samples.npz \
  --reference /path/to/jit_in256_stats.npz --output results/imf_b_adapted/fid_torch.json
```

Use an actual saved checkpoint filename. Generation saves exactly indexed RGB PNGs, an `arr_0` uint8 NHWC NPZ, and a manifest with model, CFG, steps, and seeds. Existing nonempty output directories are refused. OpenAI evaluation uses `--protocol openai`, its TensorFlow dependency, and `VIRTUAL_imagenet256_labeled.npz`. **OpenAI and torch-fidelity numbers are different protocols and must be reported separately.**

## Code and validation

`config.py` validates continuous runtime configuration; `models/registry.py` and the native/TiTok registries own model defaults. Model wrappers, losses, data pairing, training entrypoints, and evaluation are separate. [Migration and source provenance](docs/migration.md) explain compatibility with experimental checkpoints. [Reproduction status](docs/reproduction.md) distinguishes verified experiment evidence from unresolved result mappings. [Validation details](docs/validation.md) record the CPU checks and remaining GPU checks.

```bash
python -m unittest discover -s tests -v
```

CPU contract tests cover configuration, decoder gradients, loss updates, checkpoint compatibility, pairing, indexed sampling, and token-pool resume guards. They do not replace a CUDA training/evaluation run with the original large checkpoints. Upstream architectures and feature networks retain their source headers; model weights and external repositories remain subject to their upstream terms.
