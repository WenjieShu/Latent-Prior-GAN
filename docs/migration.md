# Migration and source provenance

This release reorganizes LP-GAN's continuous and TiTok experimental paths using the registry/configuration separation and reader-facing training/evaluation structure of [FDDA](https://github.com/sunset-clouds/Frechet-Distributional-Decoder-Alignment) and [GAR](https://github.com/sunset-clouds/Generation-Aware-Reconstruction).

## Public entrypoints

| Experimental name | Release entrypoint |
|:--|:--|
| `train_supervised_gan.py` | `train_lpgan.py` |
| `train_pure_gan.py` | `train_pure_gan.py` |
| `train_titok_latent_prior_gan.py` | `train_titok_lpgan.py` |
| `precompute_ar_latents.py` | `precompute_titok_real_tokens.py` |
| `precompute_latents.py` | `precompute_titok_generated_tokens.py` |
| `check_ar_pairing.py` | `check_titok_pairing.py` |
| per-experiment generation/evaluation launchers | `eval_generation.py`, `eval_fid.py` |

Machine-specific Slurm scripts, debug runs, converted JAX iMF code, unsupported prior branches, and unrelated FD post-training variants are absent from the release. Model wrappers retain the parameter nesting expected by historical checkpoints.

## Source snapshots

The continuous path comes from the audited `Latent-Prior-GAN-External` experimental snapshot, including official iMF and validated DMF/MeanFlow adapters. The TiTok path comes from the audited `FD_AR-main` snapshot: its 76-file local source matched the corresponding server source byte for byte. The torch-fidelity evaluator follows cleaned GAR. These source snapshots were audited on 2026-10-09; a snapshot is not proof of an exact historical run's Git revision.

[source_manifest.json](source_manifest.json) records selected imported paths and their **source** SHA256 hashes. Edited release files naturally differ from those source hashes. It records provenance rather than serving as a hash lock for this release. Upstream architecture headers and copyright notices have been retained. Upstream models, feature weights, and external repositories have their own terms; this cleanup does not assign them a new license.

## Compatibility and fixes

The three loss implementations and official iMF architecture/sampler preserve the audited equations. G-before-D order, detach placement, optimizer settings, DDP structure, crop choices, and normalized-latent bridges remain intact. The release changes entrypoints, validates config and CFG overrides, chooses `LOCAL_RANK` for device mapping, and provides common generation and explicit evaluation protocols.

Adapted decoder evaluation accepts full historical pipeline state dicts (`tokenizer.vae.*` or `tokenizer.model.model.*`) and tokenizer state dicts, including `module.` prefixes. It requires every decoder-side tensor and verifies shapes and available family/model metadata. A nonmatching checkpoint fails instead of silently evaluating the original decoder. This does not prove identity of original prior weights when historical metadata is absent.

Real and generated TiTok pools retain original image indices and labels. Generated pools now save the source indices used for training pairing. Streamed shards carry a configuration/checkpoint fingerprint; deterministic resume rejects changed settings and legacy shards without metadata. Legacy pools lacking the required source-image or crop metadata must be regenerated.

Checkpoint saving now includes LeCAM EMA and a completed-epoch flag. TiTok adds completed-epoch resume with its original decoder LR schedule. Historical missing state is handled as described in [reproduction.md](reproduction.md), rather than advertised as exact continuation.

## Validation

The CPU suite uses a small real diffusers VAE, genuine loss forwards/backwards with lightweight feature-network substitutes, synthetic ImageFolder/token pools, legacy checkpoint key fixtures, and official deterministic noise sampling. It checks decoder-only gradients, discriminator detach behavior, config precedence/invalid settings, checkpoint rejection, indexed sample count/pixels, and safe pool resume. Feature substitutes exercise loss plumbing; they do not verify pretrained DINO/LPIPS numerical behavior. Large-model CUDA training and FID are outside these CPU checks.
