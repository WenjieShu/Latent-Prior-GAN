"""Paper sampling defaults and checkpoint locations; no model imports."""

IMF_MODELS = {
    "iMF-B-2": dict(model_str="imfDiT_B_2", omega=8.0, t_min=0.4, t_max=0.65),
    "iMF-M-2": dict(model_str="imfDiT_M_2", omega=10.5, t_min=0.4, t_max=0.6),
    "iMF-L-2": dict(model_str="imfDiT_L_2", omega=10.5, t_min=0.4, t_max=0.6),
    "iMF-XL-2": dict(model_str="imfDiT_XL_2", omega=8.0, t_min=0.42, t_max=0.62),
}
FAMILIES = ("imf", "dmf", "meanflow", "titok")


def validate_prior_inputs(args):
    """Fail before loading a VAE or allocating a large missing prior."""
    from pathlib import Path

    checkpoint = (
        args.pretrained_imf_pytorch if args.prior_family == "imf" else args.prior_checkpoint
    )
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"Missing {args.prior_family} prior checkpoint: {checkpoint or '(unset)'}"
        )
    if args.prior_family != "imf":
        root = Path(args.prior_repo_path)
        source = "models/dmft.py" if args.prior_family == "dmf" else "src/training/network_utils.py"
        if not args.prior_repo_path or not (root / source).is_file():
            raise FileNotFoundError(
                f"Invalid native {args.prior_family} checkout: {args.prior_repo_path or '(unset)'}"
            )
    if args.prior_family == "meanflow" and (
        not args.prior_vae_checkpoint or not Path(args.prior_vae_checkpoint).is_file()
    ):
        raise FileNotFoundError(
            "MeanFlow requires its native EMA VAE snapshot (--prior_vae_checkpoint)"
        )


def model_defaults(name):
    try:
        return dict(IMF_MODELS[name])
    except KeyError as exc:
        raise ValueError(f"Unknown iMF model: {name}") from exc


def validate_imf_checkpoint_path(path):
    if not str(path).endswith(".pth"):
        raise ValueError("iMF requires an official .pth checkpoint; converted .pt is unsupported")


def resolve_continuous_defaults(args):
    if args.prior_family == "imf":
        args.pretrained_imf_pytorch = (
            args.pretrained_imf_pytorch or f"checkpoints/imf/{args.model_type}.pth"
        )
        validate_imf_checkpoint_path(args.pretrained_imf_pytorch)
        defaults = model_defaults(args.model_type)
        for field, key in [("cfg_omega", "omega"), ("cfg_t_min", "t_min"), ("cfg_t_max", "t_max")]:
            if getattr(args, field) is None:
                setattr(args, field, defaults[key])
    args.vae_type = args.vae_type or ("ema" if args.prior_family == "meanflow" else "mse")
    required = "ema" if args.prior_family == "meanflow" else "mse"
    if args.vae_type != required:
        raise ValueError(f"{args.prior_family} requires --vae_type {required}")
    return args
