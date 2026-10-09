import torch
from torch import nn

from models.discrete_tokenizer import Tokenizer


class ARLatentPriorDecoder(nn.Module):
    """Trainable TiTok decoder fed by precomputed real and AR token sequences."""

    def __init__(self, tokenizer_name, tokenizer_ckpt_path=""):
        super().__init__()
        self.tokenizer = Tokenizer(tokenizer_name, tokenizer_ckpt_path)
        self.tokenizer.train_mode()

    def decode_pair(self, real_tokens, generated_tokens):
        real_images = self.tokenizer.latent_to_image(real_tokens)
        generated_images = self.tokenizer.latent_to_image(generated_tokens)
        return real_images, generated_images

    def forward(self, real_tokens, generated_tokens):
        return self.decode_pair(real_tokens, generated_tokens)

    @property
    def last_layer(self):
        return self.tokenizer.last_layer

    def trainable_parameters(self):
        return self.tokenizer.trainable_parameters()
