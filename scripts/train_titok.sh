#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_root/scripts/torchrun.sh" train_titok_lpgan.py --tokenizer_name titok-vql32 --batch_size 2 --epochs 5 "$@"
