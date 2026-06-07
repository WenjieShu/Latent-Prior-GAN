


### Gaussian noise -> generator (with cfg and label) -> decoder 

## generator frozen or not, if not frozen, please refer to FD-loss, how to use GAN loss to update the generator
## decoder is always trainable
class LatentPriorGAN(nn.Module):
    pass