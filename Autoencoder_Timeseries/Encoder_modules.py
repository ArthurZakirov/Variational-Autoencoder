import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, encoder_params):
        super(Encoder, self).__init__()

        self.input_embedder = nn.Linear(encoder_params['input_dim'],
                                        encoder_params['hidden_dim'])
        self.encoder = nn.LSTM(encoder_params['hidden_dim'],
                               encoder_params['hidden_dim'],
                               num_layers=encoder_params['num_layers'],
                               batch_first=encoder_params['batch_first'])
        self.latent_embedder = nn.Linear(encoder_params['hidden_dim'],
                                         encoder_params['latent_dim'])

    def forward(self, x):
        x_emb = self.input_embedder(x)
        out, (h, c) = self.encoder(x_emb)
        last_timestep = out[:, -1, :].unsqueeze(1)
        z = self.latent_embedder(last_timestep)

        return z