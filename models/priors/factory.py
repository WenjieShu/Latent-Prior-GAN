"""Backend dispatch shared by training and evaluation."""


def build_prior(args):
    if args.prior_family == "imf":
        from models.imf_official import OfficialImfWrapper
        prior = OfficialImfWrapper(args.model_type, args.pretrained_imf_pytorch)
        for field in ("omega", "t_min", "t_max"):
            value = getattr(args, "cfg_" + field, None)
            if value is not None:
                setattr(prior, field, float(value))
        return prior
    if args.prior_family == "dmf":
        from .dmf import DmfPrior
        return DmfPrior(args)
    if args.prior_family == "meanflow":
        from .alphaflow import AlphaFlowPrior
        return AlphaFlowPrior(args)
    raise ValueError(f"Unsupported continuous prior: {args.prior_family}")
