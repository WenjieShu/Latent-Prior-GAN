# Migration and source provenance

This release reorganizes LP-GAN's continuous and TiTok experimental paths using the registry/configuration separation and reader-facing training/evaluation structure of [FDDA](https://github.com/sunset-clouds/Frechet-Distributional-Decoder-Alignment) and [GAR](https://github.com/sunset-clouds/Generation-Aware-Reconstruction).

## Public entrypoints

| Experimental name | Release entrypoint |
|:--|:--|
| `train_supervised_gan.py` | `train.py` (LP-GANs w/ PS, default) |
| `train_pure_gan.py` | `train.py --wo_ps` (LP-GANs w/o PS) |
| `train_titok_latent_prior_gan.py` | `train.py --config configs/titok_l32.yaml` |
| `precompute_ar_latents.py` | `precompute_titok_real_tokens.py` |
| `precompute_latents.py` | `precompute_titok_generated_tokens.py` |
| `check_ar_pairing.py` | `check_titok_pairing.py` |
| per-experiment generation/evaluation launchers | `eval_generation.py`, `eval_fid.py` |

Machine-specific Slurm scripts, debug runs, converted JAX iMF code, unsupported prior branches, and unrelated FD post-training variants are absent from the release. Model wrappers retain the parameter nesting expected by historical checkpoints.

Training now has one public entrypoint and one launcher, `scripts/train.sh`. PS means paired supervision, following the paper. Continuous and TiTok loops retain their separate experimental update and LR policies inside `train.py`. Family-specific configuration rejects incompatible options. The old TiTok `--max_steps` option is now the common `--debug_steps`; old entrypoint filenames are retained only in provenance records. Loss modules are named `paired_supervision_losses.py` and `wo_ps_losses.py`, with their equations unchanged.

## Public interfaces and configuration

Continuous and TiTok model wrappers now expose `train_mode()`, `eval_mode()`, `trainable_parameters()`, and `last_layer`. Both training forwards return `(generated_images, paired_images)`; continuous w/o PS returns `None` for the second item. TiTok still decodes paired tokens before generated tokens, preserving the experimental execution order. Continuous SD-VAE tokenizers provide `extract_latents()` and `latent_to_image()` alongside their explicit normalized `encode()`/`decode()` methods. Parameter nesting (`tokenizer.vae.*`, `tokenizer.model.model.*`, `diffusion.*`) is retained for checkpoint loading.

The old continuous `get_decoder_params()`, `encoder_eval()`, and redundant mode helpers are replaced by the public methods above. Model internals remain behind those methods. Optimizer construction, checkpoint assembly, rank checks, process-group initialization, and cleanup are shared. Continuous and TiTok update loops keep their experimental optimizer-zeroing, DDP wrapping, RNG/backend settings, detach points, and LR policies. TiTok's 7200-second process-group timeout is retained; non-master printing no longer requires modifying Python's built-in `print`.

Training help and YAML keys are backend-specific: iMF uses official `.pth` and CFG options; DMF uses native source/checkpoint and precision options; MeanFlow uses native source/checkpoint and EMA-VAE snapshot options. Previously accepted fields that had no effect on a backend are rejected. Historical checkpoint args may still contain those fields; loading saved state does not reparse them as a new training config.

Evaluation reads either continuous or TiTok training YAML. Only model and sampling fields determine evaluation; optimizer/output/PS fields are not imported into the evaluation args. Evaluation always initializes from pretrained decoder weights before optionally loading an adapted decoder checkpoint, including when the training YAML says `decoder_init: random`. TiTok evaluation resolves the generator/tokenizer pair from the registry and does not require training pools.

Unused checkpoint/LR helpers, alternative adversarial-loss functions, and the redundant dataset loader are removed. Loss equations used by the experiments remain unchanged. Ruff checks are scoped in `pyproject.toml` to project orchestration, wrappers, losses, and tests; imported architecture/feature-network code retains its upstream formatting.

## Source snapshots

The continuous path comes from the audited `Latent-Prior-GAN-External` experimental snapshot, including official iMF and validated DMF/MeanFlow adapters. The TiTok path comes from the audited `FD_AR-main` snapshot: its 76-file local source matched the corresponding server source byte for byte. The torch-fidelity evaluator follows cleaned GAR. These source snapshots were audited on 2026-10-09; a snapshot is not proof of an exact historical run's Git revision.

[source_manifest.json](source_manifest.json) records selected imported paths and their **source** SHA256 hashes. Edited release files naturally differ from those source hashes. It records provenance rather than serving as a hash lock for this release. Upstream architecture headers and copyright notices have been retained. Upstream models, feature weights, and external repositories have their own terms; this cleanup does not assign them a new license.

The README overview and pipeline figures were copied from `figs/main/lp_figure1.pdf` and `lp_figure2.pdf` in the local `_ICLR_2027__Latent_Prior_GANs (2)` paper snapshot; PNGs are rendered previews of those PDFs. Result tables reproduce that paper snapshot rather than new validation runs.

## Compatibility and fixes

The three loss implementations and official iMF architecture/sampler preserve the audited equations. G-before-D order, detach placement, optimizer settings, DDP structure, crop choices, and normalized-latent bridges remain intact. The release changes entrypoints, validates config and CFG overrides, chooses `LOCAL_RANK` for device mapping, and provides common generation and explicit evaluation protocols.

Adapted decoder evaluation accepts full historical pipeline state dicts (`tokenizer.vae.*` or `tokenizer.model.model.*`) and tokenizer state dicts, including `module.` prefixes. It requires every decoder-side tensor and verifies shapes and available family/model metadata. A nonmatching checkpoint fails instead of silently evaluating the original decoder. This does not prove identity of original prior weights when historical metadata is absent.

Real and generated TiTok pools retain original image indices and labels. Generated pools now save the source indices used for training pairing. Streamed shards carry a configuration/checkpoint fingerprint; deterministic resume rejects changed settings and legacy shards without metadata. Legacy pools lacking the required source-image or crop metadata must be regenerated.

Checkpoint saving now includes LeCAM EMA and a completed-epoch flag. TiTok adds completed-epoch resume with its original decoder LR schedule. Historical missing state is handled as described in [reproduction.md](reproduction.md), rather than advertised as exact continuation.

## Validation

The CPU suite uses a small real diffusers VAE, genuine loss forwards/backwards with lightweight feature-network substitutes, synthetic ImageFolder/token pools, legacy checkpoint key fixtures, and official deterministic noise sampling. It checks decoder-only gradients, discriminator detach behavior, config precedence/invalid settings, checkpoint rejection, indexed sample count/pixels, and safe pool resume. Feature substitutes exercise loss plumbing; they do not verify pretrained DINO/LPIPS numerical behavior. Large-model CUDA training and FID are outside these CPU checks.
