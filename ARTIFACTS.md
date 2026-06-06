# External Artifacts

This repository should remain code-focused. Do not store large checkpoints,
generated images, NPZ/NPY arrays, logs, or metric caches in git.

Expected cluster assets:

```text
/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-B-2.pt
/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-M-2.pt
/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-L-2.pt
/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-XL-2.pt
/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse/
/mnt/nfs/wenjie/dataset/imagenet_idx/
/mnt/nfs/wenjie/dataset/imagenet/val/
/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth
```

Expected V2 output roots:

```text
/mnt/nfs/wenjie/IMF-GAP_V2/checkpoints/
/mnt/nfs/wenjie/IMF-GAP_V2/results/
/mnt/nfs/wenjie/IMF-GAP_V2/logs/
```
