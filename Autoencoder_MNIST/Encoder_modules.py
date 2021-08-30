import torch
import torch.nn as nn


class CNN_Encoder(nn.Module):
    def __init__(self, height, kernel, channels, stride, padding):
        super(CNN_Encoder, self).__init__()

        self.height = height
        self.kernel = kernel
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.heights = self.get_state_shape()

        self.encoder = nn.ModuleList([
            nn.Conv2d(channels[conv],
                      channels[conv + 1],
                      kernel[conv],
                      stride[conv],
                      padding[conv])
            for conv in range(len(kernel))])

    def get_state_shape(self):
        heights = [self.height]
        height = self.height

        for conv in range(len(self.kernel)):
            height = int((height + 2 * self.padding[conv] - self.kernel[conv]) / self.stride[conv] + 1)
            heights.append(height)
        return heights

    def forward(self, x):
        for i, conv in enumerate(self.encoder):
            x = conv(x)
            x = nn.ReLU()(x)
        return x