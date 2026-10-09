# Reproduction guide

## Dependencies and inputs

Use a CUDA PyTorch environment; Python 3.10 or newer is recommended. Install `requirements.txt`. Continuous training uses SD-VAE, a frozen DINO feature backbone, VGG, ConvNeXt-S, and LPIPS projections. TiTok uses DINO but has no perceptual loss. Decoder checkpoints and pretrained priors are separate inputs.

SD-VAE source priority is `--vae_path`, then `SD_VAE_PATH` / `IMF_SD_VAE_PATH` / `HF_SD_VAE_PATH`, then `stabilityai/sd-vae-ft-{mse,ema}`. `HF_HUB_OFFLINE=1` requests the local cache. Supply the matching MSE VAE for iMF/DMF and EMA VAE for MeanFlow.

Set `DINO_CKPT_PATH` for a local `dino_deitsmall16_pretrain.pth`; otherwise the DINO loader uses the upstream Facebook URL. VGG and ConvNeXt use torchvision's weight cache. Supply the learned LPIPS projection checkpoint:

```bash
mkdir -p checkpoints/features
curl -L 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1' \
  -o checkpoints/features/vgg.pth
export LPIPS_VGG_CKPT="$PWD/checkpoints/features/vgg.pth"
```

The experimental LPIPS loader checks the copied projection checkpoint against MD5 `d507d7349b931f0638a25a48a722f98a`. Its input scaling and the fixed LPIPS/ConvNeXt mixture are preserved.

## Continuous priors

iMF uses the embedded official PyTorch inference architecture and official raw state dictionaries with `net.*` keys. The adapted pipeline keeps historical `diffusion.model.net.*` and `tokenizer.vae.*` keys. Converted iMF `.pt` priors are rejected before model allocation. Official checkpoint names/paths can be overridden; the registry supplies architecture and CFG defaults for B/M/L/XL.

DMF uses its original source checkout, not a port of the sampler. The checkout must contain `models/dmft.py` and `samplers.py`. Its native checkpoint remains `.pt`; the adapter handles the SD-VAE latent scaling bridge. The validated default CFG scale is 1.0. In this native paired-denoising adapter, `supervision_use_cfg` does not change its denoising equation; no new guidance behavior was introduced during cleanup.

```bash
pip install -r requirements-external-priors.txt
NPROC_PER_NODE=8 bash scripts/train_lpgan.sh --config configs/dmf_xl.yaml \
  --prior_repo_path /path/to/DMF --prior_checkpoint /path/to/dmf_xl.pt \
  --vae_path /path/to/sd-vae-ft-mse --dataset_dir /path/to/imagenet \
  --checkpoint_dir checkpoints/runs/dmf_xl
```

MeanFlow uses the validated AlphaFlow source/snapshot infrastructure. The checkout must contain `src/inference.py` and `src/training/network_utils.py`. Supply the model snapshot, native EMA VAE snapshot, and its matching diffusers VAE directory. `--vae_path` is also passed to the upstream loader through `ALPHAFLOW_SD_VAE_EMA_PATH`; without it the adapter follows the original local-directory convention. The adapter verifies tensor identity between the loaded native VAE and tokenizer before releasing the duplicate VAE. Native one-step trajectory sampling, CFG, and latent normalization are preserved.

```bash
NPROC_PER_NODE=8 bash scripts/train_lpgan.sh --config configs/meanflow_xl.yaml \
  --prior_repo_path /path/to/AlphaFlow --prior_checkpoint /path/to/meanflow_snapshot.pt \
  --prior_vae_checkpoint /path/to/ema_vae_snapshot.pt \
  --vae_path /path/to/sd-vae-ft-ema --dataset_dir /path/to/imagenet \
  --checkpoint_dir checkpoints/runs/meanflow_xl
```

The optional dependency file covers the audited snapshot-loader imports. Native repositories can require additional packages matching their own environment and checkpoint version. These checkouts are inputs to the release, rather than bundled copies with an invented historical revision.

## Training contracts and resume

All objectives retain AdamW betas `(0.9, 0.95)`, clipping, augmentation, adaptive adversarial weighting where applicable, DINO discriminator, LeCAM, consistency regularization, and G-before-D order. Continuous objectives preserve random crop/flip for source images. TiTok source images use the same ADM center crop as their cached tokens.

The continuous paired loss is `MSE + perceptual + adaptive_weight * disc_weight * hinge_G`. The historical `gan_weight`, `rec_weight`, and `lpips_weight` did not control that equation and are absent from the public configuration. The pure-GAN generator loss is unweighted hinge; its `disc_weight` controls the discriminator term. These experimental choices have been preserved.

Completed-epoch resume restores model, discriminator, both optimizers, epoch and step counts. New checkpoints also save LeCAM EMA. Old checkpoints without EMA restart that statistic. Continuous restored learning rates take priority over CLI values; use `--reset_resume_lr --lr ... --disc_lr ...` to change them intentionally. Some historical iMF L/XL and MeanFlow resumed runs used optimizer rates of `2e-6`, although their launch/config rates differed.

TiTok checkpoints are saved before the original epoch LR decay. Resume applies that decay once before the next epoch. Only decoder LR decays by 0.5, floored at `1e-6`; discriminator LR stays fixed. New partial-epoch diagnostic checkpoints explicitly reject training resume. Historical checkpoints lack that flag, so select known completed-epoch checkpoints. Per-rank RNG/dataloader state is not saved; resume is not guaranteed to reproduce an uninterrupted run bit for bit.

TiTok L32's audited launch used eight GPUs, batch 2 per GPU, seed 3407, initial decoder/discriminator LR `1e-5`, weight decay 0, MSE/GAN weights 1, and consistency weight 0.1. B64 follows its registered architecture/sampling defaults; the audited paper run was L32, so B64 should not be described as an independently verified experiment.

## Evaluation protocols

Generation uses sample index XOR seed and interleaved ImageNet class labels. The manifest records the actual native sampling configuration. PNGs and NPZ use rounded uint8 RGB pixels. Baseline/adapted comparisons should share model, sampling, sample count, seed, and evaluation protocol. Original large-model GPU runs remain necessary to validate numerical reproduction.

For PyTorch Inception, install `requirements-eval.txt` and use `--protocol torch_fidelity --reference /path/to/jit_in256_stats.npz`. For OpenAI Inception, additionally install a TensorFlow version compatible with the environment (the evaluator uses `tensorflow.compat.v1`):

```bash
pip install 'tensorflow>=2.13,<3'
python eval_fid.py --protocol openai \
  --samples results/titok_l32_adapted/samples.npz \
  --reference /path/to/VIRTUAL_imagenet256_labeled.npz \
  --output results/titok_l32_adapted/fid_openai.json
```

OpenAI and torch-fidelity use distinct Inception implementations and reference statistics. They are not interchangeable; filenames and result JSON identify the selected protocol.

## Historical evidence status

The source audit on 2026-10-09 found the following results. These are existing experiment evidence, not fresh runs of this cleaned release.

| Model / protocol | Baseline gFID | Adapted gFID | Evidence status |
|:--|--:|--:|:--|
| DMF XL / OpenAI | 2.4041356 | 1.9469780 | epoch 5 checkpoint and result match the paper's FID; result IS 237.0988 differs from paper IS 207.85 |
| MeanFlow XL / OpenAI | 3.4873952 | — | best imported epoch 3 checkpoint found; paper's 2.9772 result file not found |
| MeanFlow XL / torch-fidelity | 3.3397365 | 2.9443901 | separate protocol; does not establish the missing OpenAI result |
| TiTok L32 / OpenAI | 2.8065953 | 1.9841936 | best available checkpoint epoch 2 |
| TiTok L32 / torch-fidelity | 2.6458319 | 1.9026982 | separate protocol |

TiTok's formal run planned five epochs but only epochs 1–3 and termination evidence were found. Exact historical Git revisions are unavailable for the audited server snapshots. The iMF best-checkpoint/row mapping and the unresolved DMF IS/MeanFlow OpenAI evidence must be completed before claiming full paper-table reproduction.
