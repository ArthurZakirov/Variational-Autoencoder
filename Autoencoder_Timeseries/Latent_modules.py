import torch
import torch.nn as nn
import torch.distributions as td

class LatentDistribution(nn.Module):

    def __init__(self, latent_dim):
        super(LatentDistribution, self).__init__()

        self.project_to_mu = nn.Sequential(
            nn.Linear(latent_dim,
                      latent_dim),
            nn.ReLU()
        )

        self.project_to_std = nn.Sequential(
            nn.Linear(latent_dim,
                      latent_dim),
            nn.ReLU()
        )
        self.KL = None

    def sample(self, mu, std):
        eps = torch.randn(tuple(mu.shape))
        sample = mu + std * eps
        return sample

    def mode(self, mu, std):
        return mu

    def p_z_dist(self, mu, std):
        bs, dim = tuple(mu.shape)
        mu = torch.zeros((bs, dim))
        cov = torch.eye(dim).repeat(bs, 1, 1)
        p_z = td.MultivariateNormal(mu, cov)
        return p_z

    def q_phi_z_dist(self, mu, std):
        # dims
        bs, dim = tuple(mu.shape)

        # create Covariance Matrix
        dim_idx = torch.arange(dim)
        cov = torch.zeros((bs, dim, dim))
        cov[:, dim_idx, dim_idx] = std ** 2

        # return dist
        q_phi_dist = td.MultivariateNormal(mu, cov)
        return q_phi_dist

    def KL_p_q(self, p_dist, q_dist):
        kl = td.kl_divergence(p_dist, q_dist)
        kl_batch_average = kl.mean()
        return kl_batch_average

    def forward(self, z):
        bs = z.shape[0]
        z = z.view(bs, -1)

        mu = self.project_to_mu(z)
        std = torch.exp(self.project_to_std(z))

        #         if self.training:
        #             z = self.mode(mu, std).unsqueeze(1)
        #         else:
        #             z = self.sample(mu, std).unsqueeze(1)
        z = self.mode(mu, std).unsqueeze(1)

        q_dist = self.q_phi_z_dist(mu, std)
        p_dist = self.p_z_dist(mu, std)
        self.KL = self.KL_p_q(p_dist, q_dist)

        return z