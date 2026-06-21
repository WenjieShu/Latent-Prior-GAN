#!/bin/bash
#SBATCH --job-name=lpg_b_reclp_bs16
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/nfs/wenjie/Latent-Prior-GAN-latest/logs/rec_lpips_b_1ep/%x_%j.out
#SBATCH --error=/mnt/nfs/wenjie/Latent-Prior-GAN-latest/logs/rec_lpips_b_1ep/%x_%j.err

set -euo pipefail

ROOT=/mnt/nfs/wenjie/Latent-Prior-GAN-latest
VENV=/mnt/nfs/wenjie/FD-Loss/.venv
CONFIG=${ROOT}/yaml/latent_prior_gan/imf_b_nt1_gan_only.yaml
RUN_ROOT=${ROOT}/runs/rec_lpips_b_1ep
TAG=lr1e-5_dw0p65_eta0p1_g1_rec1_lpips1_bs16
NAME=lpg_b_reclp_${TAG}_1ep

export SD_VAE_PATH=/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse
export DINO_CKPT_PATH=/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HOME=/mnt/nfs/wenjie
export XDG_CACHE_HOME=/mnt/nfs/wenjie/.cache
export HF_HOME=/mnt/nfs/wenjie/hf_cache
export TORCH_HOME=/mnt/nfs/wenjie/torch_cache
export TRITON_CACHE_DIR=/mnt/nfs/wenjie/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

mkdir -p \
  "${ROOT}/logs/rec_lpips_b_1ep" \
  "${RUN_ROOT}/checkpoints/${TAG}" \
  "${RUN_ROOT}/results/${TAG}" \
  "${RUN_ROOT}/yaml" \
  "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"

cd "${ROOT}"
echo "CASE ${TAG}"
echo "START $(date)"

"${VENV}/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  post_train_rec_lpips.py \
  --config "${CONFIG}" \
  --epochs 1 \
  --batch_size 16 \
  --warmup_steps 0 \
  --lr 1e-5 \
  --disc_lr 1e-5 \
  --disc_lr_decay_steps 0 \
  --disc_weight 0.65 \
  --gan_weight 1.0 \
  --rec_weight 1.0 \
  --lpips_weight 1.0 \
  --reconstruction_noise_level 0.1 \
  --reconstruction_num_steps 1 \
  --checkpoint_dir "${RUN_ROOT}/checkpoints/${TAG}" \
  --results_dir "${RUN_ROOT}/results/${TAG}" \
  --yaml_dir "${RUN_ROOT}/yaml" \
  --saver_name_pre "${NAME}"

echo "END $(date)"
