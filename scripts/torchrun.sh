#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
launch=(--nproc_per_node="${NPROC_PER_NODE:-1}")
if [[ "${NNODES:-1}" == 1 ]]; then
  launch+=(--standalone)
else
  launch+=(--nnodes="$NNODES" --node_rank="${NODE_RANK:?Set NODE_RANK}" --master_addr="${MASTER_ADDR:?Set MASTER_ADDR}" --master_port="${MASTER_PORT:-29500}")
fi
exec "${PYTHON:-python}" -m torch.distributed.run "${launch[@]}" "$@"
