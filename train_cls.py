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

    model.train() #Set model in traininig mode

    for e in range(epoch):
        print(str(e)+'/'+str(epoch))

        # Move data to device, e.g. CPU or GPU
        # if(USE_GPU and torch.cuda.is_available()):
        #     dataset_x = dataset_x.cuda(device)
        #     dataset_y = dataset_y.cuda(device)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        #Forward
        predictions = model.forward(data_x)

        #Loss
        loss = cross_entropy2d(predictions, data_y)


        #Backwards pass
#-> How come loss, which is a scalar, is saving all the gradients found in the backwards pass?
        loss.backward()

        #Update parameters of model
        optimizer.step()


    torch.save(model, "./models/fc_cls.pkl")



def main():

    classifier = FCClassifier().float()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)# pick an optimizer.

    dataset_x = torch.tensor(np.load("./features/feats_x.npy"), dtype=torch.long)
    dataset_y = torch.tensor(np.load("./features/feats_y.npy"), dtype=torch.long)

    num_epochs = 20# your choice, try > 10

#-> I think this loop shouldnt be here but rather inside the train method
    # for epoch in range(num_epochs):
    train([dataset_x, dataset_y], classifier, optimizer, num_epochs)

if __name__ == '__main__':
    main()
