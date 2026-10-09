"""Strict adapted-tokenizer loading without loading a second frozen prior."""

from collections.abc import Mapping

import torch

from models.ckpt_load import extract_state_dict


def load_decoder_checkpoint(tokenizer, path, family, model_name):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    if not isinstance(saved, Mapping):
        saved = vars(saved)
    saved_family = checkpoint.get("prior", {}).get("family") or saved.get("prior_family")
    if saved.get("tokenizer_name", "").startswith("titok"):
        saved_family = "titok"
    if saved_family and saved_family != family:
        raise ValueError(f"Decoder prior mismatch: saved={saved_family}, requested={family}")
    field = "tokenizer_name" if family == "titok" else "model_type"
    # External runs historically stored an unused iMF model_type.
    if family in ("imf", "titok") and saved.get(field) and saved[field] != model_name:
        raise ValueError(f"Decoder model mismatch: saved={saved[field]}, requested={model_name}")
    return load_tokenizer_state(tokenizer, extract_state_dict(checkpoint, "adapted decoder"), family)


def load_tokenizer_state(tokenizer, state, family):
    expected = tokenizer.state_dict()
    selected = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("tokenizer.")
        if key in expected:
            if value.shape != expected[key].shape:
                raise ValueError(f"Decoder tensor shape mismatch: {key}: {value.shape} != {expected[key].shape}")
            if key in selected:
                raise ValueError(f"Duplicate tokenizer tensor: {key}")
            selected[key] = value
        elif key.startswith(("vae.", "model.", "mean", "std")):
            raise ValueError(f"Unexpected tokenizer tensor: {key}")
    prefixes = ("model.model.decoder.", "model.model.pixel_quantize.", "model.model.pixel_decoder.") if family == "titok" else ("vae.decoder.",)
    required = {k for k in expected if k.startswith(prefixes)}
    missing = required - selected.keys()
    if not required or missing:
        raise ValueError(f"Incomplete adapted decoder: {len(required)} expected tensors, missing={sorted(missing)[:8]}")
    tokenizer.load_state_dict(selected, strict=False)
    return {"loaded_tensors": len(selected), "decoder_tensors": len(required)}
