"""Inspect Latent Prior GAN checkpoints for train/eval compatibility."""

import argparse
from collections.abc import Mapping

import torch


REQUIRED_KEYS = {
    "model",
    "optimizer",
    "discriminator",
    "optimizer_disc",
    "epoch",
    "global_step",
    "args",
}


def _state_keys(state):
    if isinstance(state, Mapping):
        return list(state.keys())
    return []


def main():
    parser = argparse.ArgumentParser(description="Inspect a Latent Prior GAN checkpoint.")
    parser.add_argument("checkpoint", help="path to checkpoint .pth/.pth.tar")
    parser.add_argument("--strict", action="store_true", help="fail when required keys are missing")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint).__name__}")

    keys = sorted(checkpoint.keys())
    missing = sorted(REQUIRED_KEYS.difference(keys))
    model_state = checkpoint.get("model", {})
    model_keys = _state_keys(model_state)
    tokenizer_keys = [key for key in model_keys if key.startswith("tokenizer.")]
    decoder_keys = [key for key in model_keys if key.startswith("tokenizer.vae.decoder.")]

    print(f"checkpoint: {args.checkpoint}")
    print(f"top-level keys: {keys}")
    print(f"missing required keys: {missing if missing else 'none'}")
    print(f"epoch: {checkpoint.get('epoch', 'missing')}")
    print(f"global_step: {checkpoint.get('global_step', 'missing')}")
    print(f"model tensors: {len(model_keys)}")
    print(f"tokenizer tensors: {len(tokenizer_keys)}")
    print(f"decoder tensors: {len(decoder_keys)}")
    print(f"has discriminator: {checkpoint.get('discriminator') is not None}")
    print(f"has optimizer: {checkpoint.get('optimizer') is not None}")
    print(f"has optimizer_disc: {checkpoint.get('optimizer_disc') is not None}")

    if args.strict and missing:
        raise SystemExit(1)
    if args.strict and not tokenizer_keys:
        print("strict failure: checkpoint['model'] has no tokenizer.* keys")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
