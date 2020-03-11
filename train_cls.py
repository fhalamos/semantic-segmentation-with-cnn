import sys
import torch
import argparse
import numpy as np
from PIL import Image
import json
import random
from scipy.misc import toimage, imsave

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
import torchvision.transforms as transforms

from losses.loss import *
from nets.classifier import FCClassifier

from data.loader import PascalVOC
import torch.optim as optim
from utils import *

def train(dataset, model, optimizer, epoch):
    """
    TODO: Implement training for simple FC classifier.
        Input: Z-dimensional vector
        Output: label.
    """

    batch_size = 20# Can be fairly large

    data_x, data_y = dataset


    tensor_x = torch.tensor(data_x, dtype=torch.long)
    tensor_y = torch.tensor(data_y, dtype=torch.long)

    model.train() #Set model in traininig mode

    print("Epoch: "+str(epoch))

    # Move data to device, e.g. CPU or GPU
    # if(USE_GPU and torch.cuda.is_available()):
    #     dataset_x = dataset_x.cuda(device)
    #     dataset_y = dataset_y.cuda(device)



    #ALTERNATIVE 1
    # Zero out all of the gradients for the variables which the optimizer
    # will update.
    optimizer.zero_grad()

    #Forward
    predictions = model.forward(tensor_x)

    #Loss
    loss = cross_entropy2d(predictions, tensor_y)

    #Backwards pass
    #-> How come loss, which is a scalar, is saving all the gradients found in the backwards pass?
    loss.backward()

    #Update parameters of model
    optimizer.step()


    #ALTERNATIVE 2

    # tensor_dataset = data.TensorDataset(tensor_x, tensor_y)
    # loader = data.DataLoader(tensor_dataset, batch_size = batch_size, shuffle=True)

    # for t, (x,y) in enumerate(loader):

    #     # Zero out all of the gradients for the variables which the optimizer
    #     # will update.
    #     optimizer.zero_grad()

    #     #Forward
    #     predictions = model.forward(x)

    #     #Loss
    #     loss = cross_entropy2d(predictions, y)


    #     #Backwards pass
    #     #-> How come loss, which is a scalar, is saving all the gradients found in the backwards pass?
    #     loss.backward()

    #     #Update parameters of model
    #     optimizer.step()


    print(loss)


    torch.save(model, "./models/fc_cls.pkl")



def main():

    classifier = FCClassifier().float()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)# pick an optimizer.

    dataset_x = np.load("./features/feats_x.npy")
    dataset_y = np.load("./features/feats_y.npy")

    num_epochs = 20# your choice, try > 10

#-> I think this loop shouldnt be here but rather inside the train method

    for epoch in range(num_epochs):
        train([dataset_x, dataset_y], classifier, optimizer, epoch)

if __name__ == '__main__':
    main()
