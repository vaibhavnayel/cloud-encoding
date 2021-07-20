import torch
import torch.nn as nn
import torch.nn.functional as F
from perceiver_pytorch import Perceiver


class Encoder(nn.Module):
    def __init__(self,c_dim=512,depth=6):
        super().__init__()
        self.model = Perceiver(
            input_channels = 3,          # number of channels for each token of the input
            input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 64,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 1120.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = depth,                   # depth of net
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 128,            # latent dimension
            cross_heads = 2,             # number of heads for cross attention. paper said 1
            latent_heads = 6,            # number of heads for latent self attention, 8
            cross_dim_head = 64,
            latent_dim_head = 64,
            num_classes = c_dim,          # output number of classes
            attn_dropout = 0.,
            ff_dropout = 0.,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )
    def forward(self, x):
        return self.model(x)