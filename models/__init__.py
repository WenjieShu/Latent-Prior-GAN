"""Clean model entry points for IMF-GAP V2."""


def get_tokenizer():
    from models.tokenizer import Tokenizer
    return Tokenizer


def get_diffusion_model():
    from models.diffusion_pytorch import DiffusionModelPyTorch
    return DiffusionModelPyTorch


def get_gar_model():
    from models.gar_model import TokenizerFlowComposition
    return TokenizerFlowComposition


def get_gar_loss():
    from models.gar_loss import PostTrainingLoss
    return PostTrainingLoss


def get_extension_gar_loss():
    from models.generation_gan_loss import GenerationGANLoss
    return GenerationGANLoss
