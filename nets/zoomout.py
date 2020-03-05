"""
TODO: Implement zoomout feature extractor.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models



class Zoomout(nn.Module):
    def __init__(self):
        super(Zoomout, self).__init__()

        # load the pre-trained ImageNet CNN and list out the layers
        self.vgg = models.vgg11(pretrained=True)
        self.feature_list = list(self.vgg.features.children())

        #Index of conv2d layers
        self.index_layers_to_extract = [0,3,6,8,11,13,16,18]

        """
        TODO:  load the correct layers to extract zoomout features.

        When assembling hypercolumns, you may choose which layers of the main network (e.g. VGG11 or VGG16)
        to include in the hypercolumn feature.
        A reasonable default is to include only the layers directly preceding a pooling layer
        This would result in Z=1472 in the case of VGG11.
        """
    def forward(self, x):
        #a
        """
        TODO: load the correct layers to extract zoomout features.
        Hint: use F.upsample_bilinear and then torch.cat.
        """

        layer_output = x

        output = []

        #Loop over all the layers of the net
        for index, layer in enumerate(self.feature_list):

            #Apply layer
            layer_output = layer(layer_output)

            if(index in self.index_layers_to_extract):
                # print(index)
                # print(layer)
                # print(layer_output.shape)
                output.append(layer_output)



        # layers_to_extract = torch.Tensor()

        # for layer_index in self.index_layers_to_extract:
        #     print(self.feature_list[layer_index])
        #     print(self.vgg.features.children())

        # # torch.cat()

        # print("toy aca")
        # raise NotImplementedError


# Implement the 2d cross entropy, both fc and dense classifier and finish 
# the sampling features can be a good start point for this homework