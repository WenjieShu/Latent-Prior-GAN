"""Model entry points for Latent Prior GAN experiments."""


def get_tokenizer():
    from models.tokenizer import Tokenizer
    return Tokenizer


def get_diffusion_model():
    from models.diffusion_pytorch import DiffusionModelPyTorch
    return DiffusionModelPyTorch


def get_latent_prior_gan_model():
    from models.model import LatentPriorGenerator
    return LatentPriorGenerator
