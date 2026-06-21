#!/bin/bash
#SBATCH --job-name=lpg_rec_lp_dbg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --exclude=gpu-h114,gpu-h120
#SBATCH --output=/mnt/nfs/wenjie/Latent-Prior-GAN-latest/logs/debug_rec_lpips/%x_%j.out
#SBATCH --error=/mnt/nfs/wenjie/Latent-Prior-GAN-latest/logs/debug_rec_lpips/%x_%j.err

set -euo pipefail

ROOT=/mnt/nfs/wenjie/Latent-Prior-GAN-latest
PY=/mnt/nfs/wenjie/FD-Loss/.venv/bin/python

export HOME=/mnt/nfs/wenjie
export PYTHONUNBUFFERED=1
export SD_VAE_PATH=/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse
export HF_HOME=/mnt/nfs/wenjie/hf_cache
export TORCH_HOME=/mnt/nfs/wenjie/torch_cache
export XDG_CACHE_HOME=/mnt/nfs/wenjie/.cache
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${ROOT}/logs/debug_rec_lpips" \
         "${ROOT}/runs/debug_rec_lpips/checkpoints" \
         "${ROOT}/runs/debug_rec_lpips/results" \
         "${ROOT}/runs/debug_rec_lpips/yaml"

cd "${ROOT}"
echo "START $(date)"
nvidia-smi || true

"${PY}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  scripts/generated/debug_rec_lpips_entry.py \
  --model_type iMF-B-2 \
  --dataset_dir /mnt/nfs/wenjie/dataset/imagenet_idx \
  --batch_size 2 \
  --workers 0 \
  --epochs 1 \
  --debug_steps 2 \
  --warmup_steps 0 \
  --lr 1e-5 \
  --disc_lr 1e-5 \
  --disc_weight 0.65 \
  --gan_weight 1.0 \
  --rec_weight 1.0 \
  --lpips_weight 1.0 \
  --reconstruction_noise_level 0.1 \
  --reconstruction_num_steps 1 \
  --save_interval 1 \
  --log_interval 1 \
  --checkpoint_dir "${ROOT}/runs/debug_rec_lpips/checkpoints" \
  --results_dir "${ROOT}/runs/debug_rec_lpips/results" \
  --yaml_dir "${ROOT}/runs/debug_rec_lpips/yaml" \
  --saver_name_pre lpg_b_rec_lpips_eta01_debug

echo "END $(date)"
