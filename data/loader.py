import os
import collections
import json

import os.path as osp
import numpy as np
from PIL import Image
import PIL
import collections
import torch
import torchvision

from torch.utils import data
from scipy.misc import imsave
from torch.utils import data
import random 
from torch.autograd import Variable

pascal_labels = np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
        [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
        [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
        [0,192,0], [128,192,0], [0,64,128]])


class PascalVOC(data.Dataset):
    def __init__(self,root='/local/fhalamos/downloads/VOCdevkit/VOC2012', split="train", img_transform=None, label_transform=None): #'./data/VOCdevkit/VOC2012'
        self.root = root
        self.split = split
        self.n_classes = 21
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.im_height = 224
        self.im_width = 224
        self.mean_pixel = np.array([103.939, 116.779, 123.68])

        self.files = collections.defaultdict(list)
        for split in ["train", "val"]:
            file_list = tuple(open(root + '/ImageSets/Segmentation/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/JPEGImages/' + img_name + '.jpg'
        label_path = self.root + '/SegmentationClass/' + img_name + '.png'

        pil_img = Image.open(img_path).convert('RGB')
        pil_lbl = Image.open(label_path).convert("P")

        pil_img = pil_img.resize((self.im_height, self.im_width), PIL.Image.BILINEAR)
        pil_lbl = pil_lbl.resize((self.im_height, self.im_width), PIL.Image.NEAREST)

        img = np.array(pil_img)
        lbl = np.array(pil_lbl)
        lbl[lbl==255] = 0

        pil_img = 0
        pil_lbl = 0

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        return torch.from_numpy(img), torch.from_numpy(lbl).long()


def visualize(path, predicted_label):
    label = predicted_label

    label_viz = np.zeros((label.shape[0], label.shape[1], 3))

    for unique_class in np.unique(label):
        if unique_class != 0:
            indices = np.argwhere(label==unique_class)
            for idx in range(indices.shape[0]):
                label_viz[indices[idx, 0], indices[idx, 1], :] = pascal_labels[unique_class,:]

    imsave(path, label_viz)


