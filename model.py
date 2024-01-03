# Implement your model in a script called model.py

import torch
from torch import nn

myawesomemodel = nn.Sequential(
    nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
    nn.LeakyReLU(), # Leaky ReLU allows a small gradient when the unit is not active
    nn.Conv2d(32, 64, 3), # [B, 32, 26, 26] -> [B, 64, 24, 24]
    nn.LeakyReLU(), # other activation function
    nn.MaxPool2d(2),      # [B, 64, 24, 24] -> [B, 64, 12, 12]
    nn.Flatten(),        # [B, 64, 12, 12] -> [B, 9216]
    nn.Linear(64 * 12 * 12, 10), # maps the flattened vector size 9216 to size 10
)