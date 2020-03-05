import sys
import torch
import numpy as np

from torch.utils import data

from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
# impor gc

def extract_samples(zoomout, dataset):
    """
    TODO: Follow the directions in the README
    to extract a dataset of 1x1xZ features along with their labels.
    Predict from zoomout using:
         with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
    """


    print(len(dataset)) #1464 images

    for image_idx in range(len(dataset)):
        images, labels = dataset[image_idx]

        with torch.no_grad():
            zoom_feats = zoomout.forward(images.cpu().float().unsqueeze(0))



    return zoom_feats, labels


def main():
    zoomout = Zoomout().cpu().float()


    for param in zoomout.parameters():
        param.requires_grad = False

    dataset_train = PascalVOC(split = 'train')

    features, labels = extract_samples(zoomout, dataset_train)

    np.save("./features/feats_x.npy", features)
    np.save("./features/feats_y.npy", labels)


if __name__ == '__main__':
    main()
