"""Checkpoint state-dict helpers."""


def clean_state_dict(module, wrapped_prefix, clean_prefix):
    state = module.state_dict()
    return {
        (key.replace(wrapped_prefix, clean_prefix, 1) if key.startswith(wrapped_prefix) else key): value
        for key, value in state.items()
    }


def tokenizer_prefixed_state(tokenizer):
    tokenizer_state = clean_state_dict(tokenizer, "vae.decoder.module.", "vae.decoder.")
    return {f"tokenizer.{key}": value for key, value in tokenizer_state.items()}


def tokenizer_state(tokenizer):
    return clean_state_dict(
        tokenizer,
        "vae.decoder.module.",
        "vae.decoder.",
    )


def gar_model_state(model):
    return clean_state_dict(model, "tokenizer.vae.decoder.module.", "tokenizer.vae.decoder.")


def load_tokenizer_state(tokenizer, state):
    decoder_is_wrapped = hasattr(tokenizer.vae.decoder, "module")
    tokenizer.load_state_dict(_remap_tokenizer_keys(state, decoder_is_wrapped), strict=True)


def load_gar_state(model, state):
    decoder_is_wrapped = hasattr(model.tokenizer.vae.decoder, "module")
    model.load_state_dict(_remap_decoder_keys(state, decoder_is_wrapped), strict=True)


def _remap_decoder_keys(state, decoder_is_wrapped):
    remapped = {}
    for key, value in state.items():
        has_module_prefix = key.startswith("tokenizer.vae.decoder.module.")
        has_decoder_prefix = key.startswith("tokenizer.vae.decoder.")
        if decoder_is_wrapped and has_decoder_prefix and not has_module_prefix:
            key = key.replace("tokenizer.vae.decoder.", "tokenizer.vae.decoder.module.", 1)
        elif not decoder_is_wrapped and has_module_prefix:
            key = key.replace("tokenizer.vae.decoder.module.", "tokenizer.vae.decoder.", 1)
        remapped[key] = value
    return remapped


def _remap_tokenizer_keys(state, decoder_is_wrapped):
    remapped = {}
    for key, value in state.items():
        has_module_prefix = key.startswith("vae.decoder.module.")
        has_decoder_prefix = key.startswith("vae.decoder.")
        if decoder_is_wrapped and has_decoder_prefix and not has_module_prefix:
            key = key.replace("vae.decoder.", "vae.decoder.module.", 1)
        elif not decoder_is_wrapped and has_module_prefix:
            key = key.replace("vae.decoder.module.", "vae.decoder.", 1)
        remapped[key] = value
    return remapped
