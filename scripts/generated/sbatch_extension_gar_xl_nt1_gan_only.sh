#!/bin/bash
#SBATCH --job-name=extgar_xl
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/nfs/wenjie/IMF-GAP_V2/logs/slurm/%x_%j.out
#SBATCH --error=/mnt/nfs/wenjie/IMF-GAP_V2/logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/nfs/wenjie/IMF-GAP_V2}"
VENV_DIR="${VENV_DIR:-/mnt/nfs/wenjie/FD-Loss/.venv}"
CONFIG="${CONFIG:-${PROJECT_DIR}/yaml/extension_gar/imf_xl_nt1_gan_only.yaml}"

export SD_VAE_PATH="${SD_VAE_PATH:-/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse}"
export DINO_CKPT_PATH="${DINO_CKPT_PATH:-/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_DIR}/logs/slurm"
cd "${PROJECT_DIR}"

"${VENV_DIR}/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  post_train.py \
  --config "${CONFIG}"
