"""Defaults from the validated native one-step prior adapters."""

PRIOR_DEFAULTS = {
    "dmf": {
        "model_name": "DMFT-XL/2", "dmf_depth": 20, "num_steps": 1,
        "cfg_scale": 1.0, "g_min": 0.0, "g_max": 1.0, "shift": 1.0,
        "native_latent_scale": 0.18215, "required_vae_type": "mse",
    },
    "meanflow": {
        "num_steps": 1, "sigma_noise": 1.0,
        "enable_trajectory_sampling": True, "enable_consistency_sampling": False,
        "required_vae_type": "ema",
    },
}


def prior_defaults(name):
    try:
        return dict(PRIOR_DEFAULTS[name])
    except KeyError as exc:
        raise ValueError(f"Unsupported external prior: {name}") from exc
