# Release validation

Checked on 2026-10-10 on macOS / Apple Silicon in an isolated CPU environment: Python 3.9.6, torch 2.8.0, torchvision 0.23.0, NumPy 1.26.4, diffusers 0.36.0, transformers 4.57.6, accelerate 1.10.1, OmegaConf 2.3.1. Python 3.10+ and CUDA are recommended for the actual training environment. The audited TiTok GPU environment used torch 2.6.0+cu124 / torchvision 0.21.0+cu124; the cleaned release has not yet been rerun there.

`python -m unittest discover -s tests -v`: **34 tests passed**.

The suite covers strict YAML/CLI configuration, native checkpoint formats and missing-input errors, historical decoder prefixes and rejection of partial/incorrect weights, optimizer LR and LeCAM restoration, original-image/token alignment, safe streamed-pool resume, exact indexed sample packing, real diffusers VAE decoder-only gradients, and all three GAN loss updates with lightweight discriminator/perceptual substitutes.

Unified `train.py` regression checks run the real continuous w/ PS, continuous w/o PS, and TiTok training loops with tiny decoder/feature substitutes and CPU replacements for CUDA/DDP transport. They exercise real audited loss equations, AdamW updates, and checkpoint saving/loading. The continuous tests verify G-before-D order and that w/o PS never computes paired outputs. Both families produce exactly identical model, discriminator, optimizer, and LeCAM states for two uninterrupted epochs versus one epoch followed by resume, under the deterministic test setup. TiTok checks the original decoder-only LR decay. Config checks cover family-specific options and CLI overrides; resume checks reject objective changes and partial-epoch checkpoints. These do not validate multi-process transport or real pretrained networks.

A small instance of the actual iMF architecture confirms that wrapper outputs match native sampling exactly with and without CFG. Paired denoising confirms that CFG overrides reach the native equation. A small actual TiTok ImageBert/MaskGIT confirms token outputs are unchanged when seeded samples are batched differently. DMF/MeanFlow adapter checks use small native-interface substitutes and exercise their seed/latent/CFG contracts; they do not validate real native snapshots.

All six public Python entrypoints, including both continuous and TiTok training help surfaces, imported and returned `--help` successfully. Python syntax, shell `bash -n`, and local README/documentation links passed. The three loss files remain byte-identical to their audited experimental sources. The core iMF/TiTok architecture and sampling implementations were compared against the source snapshot; mathematical operations are retained.

Full CUDA/DDP training, large-checkpoint load/continuation, feature-network numerical validation, TensorFlow evaluator execution, and 50K FID/IS reproduction remain untested by this CPU pass. A macOS `torchrun --standalone` launcher smoke was stopped at rendezvous after local hostname/DNS resolution failed; no training or GPU task was started. Shell syntax and Python entrypoints passed, but a Linux CUDA launcher run is still needed.

Historical paper-result mapping limitations are recorded separately in [reproduction.md](reproduction.md).
