import torch
import torch.nn as nn
from Encoder_modules import Encoder
from Decoder_modules import Decoder_output_as_input, Decoder_repeat_latent,Decoder_concat_output_latent
from Latent_modules import LatentDistribution



class Autoencoder(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(Autoencoder, self).__init__()

        self.encoder = nn.ModuleDict({
            'AE': Encoder(encoder_params),
            'VAE': nn.Sequential(
                Encoder(encoder_params),
                LatentDistribution(encoder_params['latent_dim']))
        })
        self.decoder = nn.ModuleDict({
            'output_as_input': Decoder_output_as_input(decoder_params),
            'repeat_latent': Decoder_repeat_latent(decoder_params),
            'concat_output_and_latent': Decoder_concat_output_latent(decoder_params)
        })

        self.encoder_type = 'VAE'
        self.decoder_type = 'output_as_input'
        self.use_integrator = False

    def display_dynamics(self):
        print(f'Integrator: {self.use_integrator}')

    def display_decoder_types(self):
        print(f"current: '{self.decoder_type}'")
        print(f"options: {[dec_type for dec_type in self.decoder]}")

    def display_encoder_types(self):
        print(f"current: '{self.encoder_type}'")
        print(f"options: {[enc_type for enc_type in self.encoder]}")

    def set_decoder_type(self, decoder_type):
        self.decoder_type = decoder_type

    def set_encoder_type(self, encoder_type):
        self.encoder_type = encoder_type

    def set_dynamics(self, use_integrator):
        self.use_integrator = use_integrator

    def integrate(self, y0, v, sample_frequency):

        bs, seq_len, dim = tuple(v.shape)
        dt = torch.ones(seq_len - 1) / sample_frequency
        y_total = list()
        for dim in range(dim):
            y_dim = list()
            for ts in range(0, seq_len):
                if ts == 0:
                    y = y0[:, dim]
                    y_dim.append(y)
                elif ts >= 1:
                    y = v[:, :ts, dim] @ dt[:ts] + y0[:, dim]
                    y_dim.append(y)

            y_dim = torch.stack(y_dim, dim=1)

            y_total.append(y_dim)
        y_total = torch.stack(y_total, dim=2)
        y_total

        return y_total

    def forward(self, x):

        z = self.encoder[self.encoder_type](x)
        y = self.decoder[self.decoder_type](x, z)

        if self.use_integrator:
            y = self.integrate(x[:, -1, :], y, 2)

        if self.encoder_type == 'VAE':
            return y, self.encoder['VAE'][1].KL
        else:
            return y, torch.tensor([0])