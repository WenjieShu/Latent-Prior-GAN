#!/bin/bash
#SBATCH --job-name=lpg_b_eval
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=/mnt/nfs/wenjie/Latent-Prior-GAN/logs/slurm/%x_%j.out
#SBATCH --error=/mnt/nfs/wenjie/Latent-Prior-GAN/logs/slurm/%x_%j.err

set -euo pipefail

IMF_GAP="/mnt/nfs/wenjie/IMF-GAP_V1"
GEN_PY="/mnt/nfs/wenjie/FD-Loss/.venv/bin/python"
TF_PY="/mnt/nfs/wenjie/imf_tf_eval/.venv/bin/python"
OUT_DIR="/mnt/nfs/wenjie/IMF-GAP_V1/results/latent_prior_gan_b_nt1_gan_only_ep10_20260608_gfid_nocfg"
CKPT="/mnt/nfs/wenjie/Latent-Prior-GAN/checkpoints/imf_b_nt1_gan_only/checkpoint-latent_prior_gan_iMF-B-2_nt1_gan_only_nocfg-epoch010.pth.tar"
NPZ="${OUT_DIR}/Generated_noCFG_iMF-B-2.npz"

export HOME="/mnt/nfs/wenjie"
export PYTHONUNBUFFERED=1
export SD_VAE_PATH="${SD_VAE_PATH:-/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse}"
export HF_HOME="${HF_HOME:-/mnt/nfs/wenjie/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/mnt/nfs/wenjie/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/nfs/wenjie/.cache}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="${IMF_GAP}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}" "/mnt/nfs/wenjie/Latent-Prior-GAN/logs/slurm"
cd "${IMF_GAP}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUT_DIR}"
date
nvidia-smi || true

if [ ! -f "${NPZ}" ]; then
  "${GEN_PY}" - <<PY
import os
import numpy as np
from scripts.eval_gfid_imf_style import load_post_trained_model, generate_images

class Args:
    post_train_ckpt = "${CKPT}"
    imf_pytorch_ckpt = "/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-B-2.pt"
    model_type = "iMF-B-2"

diff_model, tokenizer, device = load_post_trained_model(Args())
images = generate_images(
    diff_model,
    tokenizer,
    device,
    num_samples=50000,
    batch_size=64,
    use_cfg=False,
)
os.makedirs("${OUT_DIR}", exist_ok=True)
np.savez("${NPZ}", arr_0=images)
print(f"Saved generated images to ${NPZ}")
PY
else
  echo "NPZ already exists: ${NPZ}"
fi

"${TF_PY}" -u scripts/compute_fid.py \
  --npz_dir "${OUT_DIR}" \
  --model_type iMF-B-2 \
  --evaluator openai \
  --ref_npz /mnt/nfs/wenjie/IMF-GAP_V1/metric/VIRTUAL_imagenet256_labeled.npz \
  --output_json "${OUT_DIR}/fid_results_openai.json"

date
