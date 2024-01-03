# Before any training can start, you should identify what corruption that
# we have applied to the MNIST dataset to create the corrupted version.

# Implement your data setup in a script called data.py. The data was saved
# using torch.save, so to load it you should use torch.load.

import torch
import numpy
from torch import nn

def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_data, train_labels = [], []
    path = '/data'

    for i in range(5):
        train_data.append(torch.load(path + f'train_images_{i}.pt'))
        train_labels.append(torch.load(path + f'train_target_{i}.pt'))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(path + "test_images.pt")
    test_labels = torch.load(path + "test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (
        torch.utils.data.TensorDataset(train_data, train_labels),
        torch.utils.data.TensorDataset(test_data, test_labels)
    )


