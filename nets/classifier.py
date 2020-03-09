"""
   Here you will implement a relatively shallow neural net classifier on top of the hypercolumn (zoomout) features.
   You can look at a sample MNIST classifier here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from .zoomout import *
import numpy as np
from torchvision import transforms

class FCClassifier(nn.Module):
    """
        Fully connected classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, n_classes=21):
        super(FCClassifier, self).__init__()
        """
        TODO: Implement a fully connected classifier.
        """
        # You will need to compute these and store as *.npy files
        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))

    def forward(self, x):
        # normalization
        x = (x - self.mean)/self.std
        raise NotImplementedError


class DenseClassifier(nn.Module):
    """
        Convolutional classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, fc_model, n_classes=21):
        super(DenseClassifier, self).__init__()
        """
        TODO: Convert a fully connected classifier to 1x1 convolutional.
        """

        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))

        # You'll need to add these trailing dimensions so that it broadcasts correctly.
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(mean, -1), -1))
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(std, -1), -1))

    def forward(self, x):
        """
        Make sure to upsample back to 224x224 --take a look at F.upsample_bilinear
        """

        # normalization
        x = (x - self.mean)/self.std
        raise NotImplementedError
