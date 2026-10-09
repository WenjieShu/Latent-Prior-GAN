"""Checkpoint helpers that avoid duplicating frozen external priors."""


def loss_ema_state(loss):
    ema = getattr(loss, "lecam_ema", loss)
    return {key: getattr(ema, key) for key in ("logits_real_ema", "logits_fake_ema")}


def restore_loss_ema(loss, checkpoint):
    if "loss_ema" in checkpoint:
        ema = getattr(loss, "lecam_ema", loss)
        for key, value in checkpoint["loss_ema"].items():
            if key not in ("logits_real_ema", "logits_fake_ema"):
                raise ValueError(f"Unknown loss EMA key: {key}")
            setattr(ema, key, value)


def validate_epoch_resume(checkpoint):
    if checkpoint.get("epoch_complete") is False:
        raise ValueError("Partial-epoch debug checkpoints support evaluation only; resume from a completed epoch")


def advance_decoder_lr(optimizer, decay, minimum):
    for group in optimizer.param_groups:
        group["lr"] = max(minimum, group["lr"] * decay)


def model_state_for_checkpoint(model, prior_family):
    state = model.state_dict()
    if prior_family == "imf":
        return state
    return {key: value for key, value in state.items() if not key.startswith("diffusion.")}


def load_model_state(model, state, prior_family):
    if prior_family == "imf":
        model.load_state_dict(state)
        return
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.startswith("diffusion.")]
    if unexpected or missing:
        raise RuntimeError(
            f"External-prior decoder checkpoint mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )


def validate_external_prior_identity(checkpoint, args):
    if args.prior_family == "imf":
        return
    saved = checkpoint.get("prior")
    if not isinstance(saved, dict):
        raise RuntimeError("External-prior checkpoint is missing prior metadata")
    if saved.get("family") != args.prior_family:
        raise RuntimeError(
            f"Resume prior family mismatch: saved={saved.get('family')} "
            f"requested={args.prior_family}"
        )
    saved_checkpoint = saved.get("checkpoint")
    if saved_checkpoint and args.prior_checkpoint:
        import os
        if os.path.realpath(saved_checkpoint) != os.path.realpath(args.prior_checkpoint):
            raise RuntimeError(
                "Resume external-prior checkpoint differs from the saved run: "
                f"saved={saved_checkpoint} requested={args.prior_checkpoint}"
            )


def external_prior_metadata(args):
    if args.prior_family == "imf":
        return {}
    return {
        "prior": {
            "family": args.prior_family,
            "checkpoint": args.prior_checkpoint,
            "vae_checkpoint": args.prior_vae_checkpoint,
            "dtype": args.prior_dtype,
        }
    }
