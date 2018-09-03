#!/usr/bin/env python3
"""Provide CNN-based models.
"""
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size,
                 sequence_length):
        super(CNN, self).__init__()

        self.num_layers = num_layers
        
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size

            self.add_module(f'layer{l}', nn.Sequential(
                nn.Conv1d(in_channels=in_size,
                          out_channels=hidden_size,
                          kernel_size=5,
                          padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU()))

        self.fc = nn.Linear(sequence_length * hidden_size, output_size)


    def forward(self, input):
        output = input.permute(0, 2, 1)

        for l in range(self.num_layers):
            layer = getattr(self, f'layer{l}')
            output = layer(output)

        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output
