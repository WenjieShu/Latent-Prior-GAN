#!/bin/bash
#SBATCH --job-name=lpg_b
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/nfs/wenjie/Latent-Prior-GAN/logs/slurm/%x_%j.out
#SBATCH --error=/mnt/nfs/wenjie/Latent-Prior-GAN/logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/nfs/wenjie/IMF-GAP_V2}"
VENV_DIR="${VENV_DIR:-/mnt/nfs/wenjie/FD-Loss/.venv}"
CONFIG="${CONFIG:-${PROJECT_DIR}/yaml/latent_prior_gan/imf_b_nt1_gan_only.yaml}"

export SD_VAE_PATH="${SD_VAE_PATH:-/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse}"
export DINO_CKPT_PATH="${DINO_CKPT_PATH:-/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HOME="/mnt/nfs/wenjie"
export XDG_CACHE_HOME="/mnt/nfs/wenjie/.cache"
export HF_HOME="/mnt/nfs/wenjie/hf_cache"
export TORCH_HOME="/mnt/nfs/wenjie/torch_cache"
export TRITON_CACHE_DIR="/mnt/nfs/wenjie/triton_cache"

mkdir -p "/mnt/nfs/wenjie/Latent-Prior-GAN/logs/slurm" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"
cd "${PROJECT_DIR}"

"${VENV_DIR}/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  post_train.py \
  --config "${CONFIG}"
