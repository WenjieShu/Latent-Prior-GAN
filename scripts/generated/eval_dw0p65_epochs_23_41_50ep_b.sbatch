#!/bin/bash
#SBATCH --job-name=lpg_b_dw065_eval_23_41
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --exclude=gpu-h114,gpu-h120
#SBATCH --array=23-41%2
#SBATCH --output=/mnt/nfs/wenjie/Latent-Prior-GAN-latest/logs/no_warmup_dw0p65_50ep/%x_%A_%a.out
#SBATCH --error=/mnt/nfs/wenjie/Latent-Prior-GAN-latest/logs/no_warmup_dw0p65_50ep/%x_%A_%a.err

set -euo pipefail

TAG="lr1e-5_dw0p65"
EPOCH_NUM="${SLURM_ARRAY_TASK_ID}"
EPOCH_PAD=$(printf "%03d" "${EPOCH_NUM}")

ROOT=/mnt/nfs/wenjie/Latent-Prior-GAN-latest
RUN_ROOT=${ROOT}/runs/no_warmup_dw0p65_50ep
IMF_GAP=/mnt/nfs/wenjie/IMF-GAP_V1
GEN_PY=/mnt/nfs/wenjie/FD-Loss/.venv/bin/python
TF_PY=/mnt/nfs/wenjie/imf_tf_eval/.venv/bin/python

CKPT="${RUN_ROOT}/checkpoints/${TAG}/checkpoint-lpg_b_dw065_50ep-epoch${EPOCH_PAD}.pth.tar"
OUT_DIR="${RUN_ROOT}/eval_openai/${TAG}/ep${EPOCH_PAD}"
NPZ="${OUT_DIR}/Generated_noCFG_iMF-B-2.npz"

export HOME=/mnt/nfs/wenjie
export PYTHONUNBUFFERED=1
export SD_VAE_PATH=/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse
export HF_HOME=/mnt/nfs/wenjie/hf_cache
export TORCH_HOME=/mnt/nfs/wenjie/torch_cache
export XDG_CACHE_HOME=/mnt/nfs/wenjie/.cache
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="${IMF_GAP}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}" "${ROOT}/logs/no_warmup_dw0p65_50ep"
cd "${IMF_GAP}"

echo "CASE ${TAG} ep${EPOCH_PAD}"
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUT_DIR}"
echo "START $(date)"

if [ ! -f "${CKPT}" ]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 2
fi

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

echo "END $(date)"
