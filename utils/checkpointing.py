"""Checkpoint state-dict helpers."""


def clean_state_dict(module, wrapped_prefix, clean_prefix):
    state = module.state_dict()
    return {
        (key.replace(wrapped_prefix, clean_prefix, 1) if key.startswith(wrapped_prefix) else key): value
        for key, value in state.items()
    }


def tokenizer_prefixed_state(module):
    tokenizer_state = clean_state_dict(module.tokenizer, "vae.decoder.module.", "vae.decoder.")
    return {f"tokenizer.{key}": value for key, value in tokenizer_state.items()}


def latent_decoder_state(module):
    return clean_state_dict(
        module,
        "tokenizer.vae.decoder.module.",
        "tokenizer.vae.decoder.",
    )


def gar_model_state(model):
    return clean_state_dict(model, "tokenizer.vae.decoder.module.", "tokenizer.vae.decoder.")


def load_decoder_state(module, state):
    decoder_is_wrapped = hasattr(module.tokenizer.vae.decoder, "module")
    module.load_state_dict(_remap_decoder_keys(state, decoder_is_wrapped), strict=True)


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
