"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, encoder=None):
        super().__init__()

        print("Creating branched erfnet with {} classes".format(num_classes))

        self.encoder = erfnet.Encoder(sum(num_classes), input_channels) if encoder is None else encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print("Initialize last layer with size: ", output_conv.weight.size())
            print("*************************")
            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2 : 2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2 : 2 + n_sigma].fill_(1)

    def forward(self, input):
        output = self.encoder(input)
        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)
