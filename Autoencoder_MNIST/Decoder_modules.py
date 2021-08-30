import torch
import torch.nn as nn


class CNN_Decoder(nn.Module):
    def __init__(self, height, kernel, channels, stride, padding):
        super(CNN_Decoder, self).__init__()

        self.height = height
        self.kernel = kernel
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.reversed_heights = self.get_reversed_state_shape()

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(channels[conv + 1],
                               channels[conv],
                               kernel[conv],
                               stride[conv],
                               padding[conv])
            for conv in reversed(range(len(kernel)))])

    def get_reversed_state_shape(self):
        heights = [self.height]
        height = self.height

        for conv in range(len(self.kernel)):
            height = int((height + 2 * self.padding[conv] - self.kernel[conv]) / self.stride[conv] + 1)
            heights.append(height)

        reversed_heights = list(reversed(heights))[1:]
        return reversed_heights

    def forward(self, x):

        for i, deconv in enumerate(self.decoder):
            output_heights = [self.reversed_heights[i], self.reversed_heights[i]]
            x = deconv(x, output_heights)

        x = nn.Tanh()(x)
        return x