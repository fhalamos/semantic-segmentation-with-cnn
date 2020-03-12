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
from nets.zoomout import Zoomout
from nets.classifier import FCClassifier, DenseClassifier

from data.loader import *
import torch.optim as optim
from utils import *

def train(args, zoomout, model, train_loader, optimizer, epoch):
    count = 0


    model = model.to(device=device)

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        if(USE_GPU and torch.cuda.is_available()):
            images = images.cuda(device)
            labels = labels.cuda(device)


        #Generate tensor of features (hypercolumns) of every image
        images_features = []
        for image in images:
            with torch.no_grad():
                if not (USE_GPU and torch.cuda.is_available()):
                    image = image.cpu()
                z = zoomout.forward(image.float().unsqueeze(0))#.cpu()
            images_features.append(z)       
        
        features = torch.cat(images_features, dim = 0)

        features = features.to(device=device)

        model.train() #Set model in train mode

        optimizer.zero_grad() #zero out all of the gradients 

        predicts = model.forward(features)

        loss = cross_entropy2d(predicts, labels)

        loss.backward()

        optimizer.step()

        if batch_idx % 20 == 0:
            count = count + 1
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data.item()))

        if batch_idx % 20 == 0:
            """
            Visualization of results.
            """
            pred = predicts[0,:,:,:]
            gt = labels[0,:,:].data.cpu().numpy().squeeze()
            im = images[0,:,:,:].data.cpu().numpy().squeeze()
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            _, pred_mx = torch.max(pred, 0)
            pred = pred_mx.data.cpu().numpy().squeeze()
            image = Image.fromarray(im.astype(np.uint8), mode='RGB')

            image.save("./imgs/im_" + str(count) + "_" + str(epoch) + "_.png")
            visualize("./lbls/pred_" + str(count) + "_" + str(epoch) + ".png", pred)
            visualize("./lbls/gt_" + str(count) + "_" + str(epoch) + ".png", gt)

    # Make sure to save your model periodically
    torch.save(model, "./models/full_model.pkl")

def val(args, zoomout, model, val_loader):
    # modified from https://github.com/wkentaro/pytorch-fcn/blob/master/examples/voc/evaluate.py
    model = model.to(device=device)

    model.eval()
    print("Validating...")
    label_trues, label_preds = [], []

    for batch_idx, (data, target) in enumerate(val_loader):

        data, target = data.float(), target.float()

        data = data.to(device=device)
        target = target.to(device=device)

        score = model.forward(zoomout(data))

        _, pred = torch.max(score, 1)#0
        lbl_pred = pred.data.cpu().numpy().astype(np.int64)
        lbl_true = target.data.cpu().numpy().astype(np.int64)

        for _, lt, lp in zip(_, lbl_true, lbl_pred):
            label_trues.append(lt)
            label_preds.append(lp)

    n_class = 21
    metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))



USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)


def main():

    # You can add any args you want here
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='./models/fc_cls.pkl', help='Path to the saved model')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,  help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, help='Learning Rate')

    args = parser.parse_args()

    zoomout = Zoomout().float()
    zoomout = zoomout.to(device=device)


    # we will not train the feature extractor
    for param in zoomout.parameters():
        param.requires_grad = False

    fc_classifier = torch.load(args.model_path)
    fc_classifier = fc_classifier.to(device=device)

    classifier = DenseClassifier(fc_model=fc_classifier).float()
    classifier = classifier.to(device=device)

    optimizer = optim.Adam(classifier.parameters(), lr=args.l_rate)# Start in range [1e-3, 1e-4].

    dataset_train = PascalVOC(split = 'train')
    dataset_val = PascalVOC(split = 'val')

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4)

    for epoch in range(args.n_epoch):
        print(epoch)
        train(args, zoomout, classifier, train_loader, optimizer, epoch)
        val(args, zoomout, classifier, val_loader)

if __name__ == '__main__':
    main()
