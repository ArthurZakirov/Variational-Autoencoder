import torch
import torch.nn as nn
import torch.distributions as td


class Latent(nn.Module):

    def __init__(self, hidden_dim, latent_dim):
        super(Latent, self).__init__()

        self.project_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.project_to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.KL = torch.tensor([0])
        self.monte_carlo_samples = 10

    def sample(self, mu, logvar):
        var = torch.exp(logvar)
        std = torch.sqrt(var)

        eps = torch.randn_like(mu)
        z = mu + std * eps
        return z

    def KL_divergence_prior_std_normal(self, mu, logvar):
        KL = -0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)).sum(dim=1)
        return KL

    def q_z_dist(self, mu, logvar):
        var = torch.exp(logvar)
        cov = torch.diag_embed(var)
        return td.MultivariateNormal(mu, cov)

    def p_z_dist(self, mu, logvar):
        mu_prior = torch.zeros_like(mu)
        var_prior = torch.ones_like(logvar)
        cov_prior = torch.diag_embed(var_prior)
        return td.MultivariateNormal(mu_prior, cov_prior)

    def KL_divergence(self, mu, logvar):
        p_dist = self.p_z_dist(mu, logvar)
        q_dist = self.q_z_dist(mu, logvar)

        KL = td.kl_divergence(q_dist, p_dist)
        return KL

    def forward(self, h, monte_carlo_samples):

        mu = self.project_to_mu(h)
        logvar = self.project_to_logvar(h)

        if self.training:
            z_samples = [self.sample(mu, logvar)
                         for s in range(monte_carlo_samples)]
        else:
            z_samples = mu

        self.KL = self.KL_divergence_prior_std_normal(mu, logvar)

        return z_samples