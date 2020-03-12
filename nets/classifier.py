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


        #Save means and stds
        features = np.load("./features/feats_x.npy")        
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)

        np.save("./features/mean.npy", means)
        np.save("./features/std.npy", stds)  

        self.mean = torch.Tensor(means)#np.load("./features/mean.npy"))
        self.std = torch.Tensor(stds)#np.load("./features/std.npy"))

#-> Should this fcc classifier have convolutions? No right? The input is not images

        self.fc1 = nn.Linear(1472,512)
        self.fc2 = nn.Linear(512,21)

        # self.dropout1 = nn.Dropout2d(0.25)


    def forward(self, x):
        # normalization
        x = (x - self.mean)/(self.std+ 1e-5)

        x = self.fc1(x)
        x = F.relu(x)

        # x = self.dropout1(x)

        x = self.fc2(x)
        # x = F.relu(x)

        # probs = F.softmax(x, dim=0)

        return x



USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DenseClassifier(nn.Module):
    """
        Convolutional classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, fc_model, n_classes=21):
        super(DenseClassifier, self).__init__()


        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))

        # You'll need to add these trailing dimensions so that it broadcasts correctly.
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(self.mean, -1), -1))
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(self.std, -1), -1))        


        self.mean = self.mean.to(device=device)
        self.std = self.std.to(device=device)
        


        #Convert a fully connected classifier to 1x1 convolutional.               
        for index, fc_layer in enumerate(fc_model.children()):

            fc_layer_params = fc_layer.state_dict()
            dim_out, dim_in = fc_layer_params["weight"].shape    
            
            #Declare the convolution layer
            convolution = nn.Conv2d(dim_in, dim_out, 1,1)

            #Load the weights
            convolution.load_state_dict(
                {"weight":fc_layer_params["weight"].view(dim_out, dim_in, 1,1),
                "bias":fc_layer_params["bias"]})

            #Set the convolution layers to dense classifier
            setattr(self, "conv"+str(index+1), convolution)



    def forward(self, x): #x.shape = (n,1472,224,224)
        """
        Make sure to upsample back to 224x224 --take a look at F.upsample_bilinear
        """
        # normalization

        x = (x - self.mean)/(self.std+ 1e-5)

        x = self.conv1(x)
        x = F.relu(x)
        

        x = self.conv2(x)
        x = F.relu(x)
    
        return x