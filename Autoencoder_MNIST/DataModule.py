import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from torchvision import datasets as torch_datasets
from torchvision import transforms

class DataModule(pl.LightningDataModule):

    def __init__(self, data_params):
        super(DataModule, self).__init__()

        self.batch_size = data_params['batch_size']
        self.split = data_params['split']

    def setup(self):
        dataset = torch_datasets.MNIST('MNIST', train=True, download=True, transform=transforms.ToTensor())
        train_len = int(self.split[0] * len(dataset))
        val_len = int(self.split[1] * len(dataset))
        test_len = len(dataset) - train_len - val_len
        split_lens = [train_len, val_len, test_len]

        (self.train_dataset,
         self.val_dataset,
         self.test_dataset) = random_split(dataset, split_lens)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
