import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm.notebook import tqdm

from Autoencoder import CNN_VAE


def main():

    from torchvision import datasets
    dataset = datasets.MNIST('MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    #####################################################################################
    ###     MODEL     ###################################################################
    #####################################################################################
    height = 28
    kernel = [3, 3, 7]
    channels = [1, 32, 64, 128]
    stride = [2, 2, 2]
    padding = [1, 1, 0]
    hidden_dim = 128
    latent_dim = 128

    model = CNN_VAE(height, kernel, channels, stride, padding, hidden_dim, latent_dim)

    ###############################################################################################
    ###      TRAINING     #########################################################################
    ###############################################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    beta = 1
    epochs = 10
    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(tqdm(data_loader)):
            NLL, KL = model.train_loss(x, x)
            loss = NLL + KL
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()

            if (batch_idx == 0) or (batch_idx % 20 == 0):
                print(f'NLL = {NLL.detach().item():10.2f}', end=' ')
                print(f'KL = {KL.detach().item():10.2f}')

        scheduler.step()

    #######################################################################################
    ###     VISUALIZATION     #############################################################
    #######################################################################################

    batch_sample = 24


    # Original sample
    image = x[batch_sample].detach().squeeze()
    plt.figure(figsize = (3,3))
    plt.imshow(image, cmap='gray', vmin=0, vmax=1);

    # Predicted Sample
    model.eval()
    x_rec = model.predict(x)
    image = x_rec[batch_sample].detach().squeeze()
    plt.figure(figsize = (3,3))
    plt.imshow(image, cmap='gray', vmin=0, vmax=1);

    # Generated Samples
    model.train()
    monte_carlo_samples = 5
    fig, (axes) = plt.subplots(1, monte_carlo_samples, figsize = (15,15))
    x_rec = model.sample(monte_carlo_samples)

    for i, ax in enumerate(axes):
        image = x_rec[i][batch_sample].detach().squeeze()
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)


if __name__ == '__main__':
    main()