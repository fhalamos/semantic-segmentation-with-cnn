import sys
import torch
import numpy as np

from torch.utils import data

from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
# impor gc
import random

def extract_samples(zoomout, dataset):

    print(len(dataset)) #1464 images

    samples_features = []
    samples_labels = []


    for image_idx in range(len(dataset)):
        print(image_idx)
        images, labels = dataset[image_idx]

        #Get hypercolumn features
        with torch.no_grad():
            features = zoomout.forward(images.cpu().float().unsqueeze(0))

        #Total different classes in this image
        classes = labels.unique()

        #Sample 3 pixels per image from each class
        for c in classes:

            #Indices of pixels with class c
            pixels_indices_for_c = np.argwhere(labels.numpy()==c.numpy())

            #Get 3 random pixels
            for i in range(0,3):   
                random_index = random.randint(0,len(pixels_indices_for_c)-1)
                y, x = pixels_indices_for_c[random_index]                

                random_feature = features[0,range(1472),y,x]
                random_label = labels[y,x]

                samples_features.append(random_feature.detach().clone().numpy())
                samples_labels.append(random_label.detach().clone().numpy())

    return np.asarray(samples_features), np.asarray(samples_labels)


def main():

    zoomout = Zoomout().cpu().float()

    for param in zoomout.parameters():
        param.requires_grad = False

    dataset_train = PascalVOC(split = 'train')

    features, labels = extract_samples(zoomout, dataset_train)

    np.save("./features/feats_x.npy", features)
    np.save("./features/feats_y.npy", labels)

    #Save means and stds
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    np.save("./features/mean.npy", means)
    np.save("./features/std.npy", stds)  


if __name__ == '__main__':
    main()
