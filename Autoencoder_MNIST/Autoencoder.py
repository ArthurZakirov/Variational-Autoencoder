import torch
import torch.nn as nn
import torch.distributions as td
from Encoder_modules import CNN_Encoder
from Decoder_modules import CNN_Decoder
from Latent_modules import Latent


class CNN_VAE(nn.Module):

    def __init__(self, height, kernel, channels, stride, padding, hidden_dim, latent_dim):
        super(CNN_VAE, self).__init__()

        self.encoder = CNN_Encoder(height, kernel, channels, stride, padding)
        self.latent = Latent(hidden_dim, latent_dim)
        self.decoder = CNN_Decoder(height, kernel, channels, stride, padding)

    def p_y_xz_dist(self, mu_y):
        var_y = torch.exp(nn.Parameter(torch.tensor([0.])))
        p_y_xz_dist = td.Normal(mu_y, var_y)
        return p_y_xz_dist

    def predict(self, x, monte_carlo_samples=1):
        x_enc = self.encoder(x)
        z = self.latent(x_enc.squeeze(), monte_carlo_samples)
        mu_y = self.decoder(z.view(z.shape[0], z.shape[1], 1, 1))
        p_y_dist = self.p_y_xz_dist(mu_y)
        y_mean = p_y_dist.mean
        return y_mean

    def sample(self, x, monte_carlo_samples=1):
        x_enc = self.encoder(x)
        z_samples = self.latent(x_enc.squeeze(), monte_carlo_samples)
        mu_y_samples = [self.decoder(z_sample.view(z_sample.shape[0], z_sample.shape[1], 1, 1))
                        for z_sample in z_samples]
        return mu_y_samples

    def train_loss(self, x, y, monte_carlo_samples=5):
        x_enc = self.encoder(x)
        z_samples = self.latent(x_enc.squeeze(), monte_carlo_samples)  # [mc_samples][bs, latent_dim]
        mu_y_samples = [self.decoder(z_sample.view(z_sample.shape[0], z_sample.shape[1], 1, 1))
                        # [mc_samples][bs, 1, H, H]
                        for z_sample in z_samples]

        E_log_p_y_xz = torch.stack([self.p_y_xz_dist(mu_y_sample).log_prob(y)
                                    for mu_y_sample in mu_y_samples],
                                   dim=0).mean(dim=0).sum(dim=[1, 2, 3])
        KL_q_zx_p_z = self.latent.KL

        NLL_batch = - E_log_p_y_xz.mean()
        KL_batch = KL_q_zx_p_z.mean()

        return NLL_batch, KL_batch