from torch import nn


class Tokenizer(nn.Module):
    def __init__(self, tokenizer_name, tokenizer_ckpt_path=""):
        super(Tokenizer, self).__init__()
        if tokenizer_name.startswith("titok"):
            from models.titok.tokenizer import TiTokTokenizer

            self.model = TiTokTokenizer(tokenizer_name, tokenizer_ckpt_path)
        else:
            raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")

    def train_mode(self):
        return self.model.train_mode()

    def eval_mode(self):
        return self.model.eval_mode()

    def trainable_parameters(self):
        return self.model.trainable_parameters()

    @property
    def last_layer(self):
        return self.model.last_layer

    def standard_reconstruction(self, x):
        return self.model.standard_reconstruction(x)

    def extract_latents(self, x):
        return self.model.extract_latents(x)

    def latent_to_image(self, latents):
        return self.model.latent_to_image(latents)

    def forward(self, x):
        return self.model.standard_reconstruction(x)
