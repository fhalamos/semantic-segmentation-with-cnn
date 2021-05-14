# Semantic Segmentation with Convolutional Networks

In this project, we focus purely on labeling each pixel with one of a fixed set of categories - a process often called semantic segmentation, as the labels partition the image into regions for each category.  For this assignment, we will ignore the distinction between different instances of the same category (there are other approaches for achieving both category-level and instance-level segmentation).

You will train a convolutional neural network (CNN)-based architecture on a widely used benchmark dataset for semantic segmentation (PASCAL VOC).  You will be implementing a variant of a hypercolumn architecture, taking a CNN pre-trained on ImageNet and adapting it to the semantic segmentation task.

## Pytorch

This project will be done in PyTorch, a popular Python deep learning framework.  Go to https://pytorch.org/ and follow the instructions to download the most recent stable build for your machine -- under CUDA select "None" (unless you have access to an NVidia GPU, which is not necessary for this project.)

To familiarize yourself with PyTorch, it is recommended that you look at some of the tutorials -- this "60 minute blitz" should help with the basics:
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

## Downloading the Dataset

Download the training and validation data through this link:
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
Extract it into ./data.

## Hypercolumn / Zoom-out Model

The provided code for this project is adapted from the following paper:

Mohammadreza Mostajabi, Payman Yadollahpour, Gregory Shakhnarovich,
"Feedforward semantic segmentation with zoom-out features",
https://arxiv.org/abs/1412.0774

which is a roughly equivalent design to the hypercolumn architecture discussed earlier in lecture:

Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik,
"Hypercolumns for Object Segmentation and Fine-grained Localization",
https://arxiv.org/abs/1411.5752

Note that instead of using superpixels (as in the zoom-out paper), we will simply sample hypercolumn-like descriptors at individual pixel locations, and make per-pixel predictions.  Specifically, our network will take the activations of a pre-trained classification network (VGG11) and upsample them to be the same resolution as in the input image, concatenating these activations into the final feature descriptor.

For example, the activation of layer convX may be of size H/8 x W/8 x D as the network has some number of pooling layers before that point.  We would upsample these to be H x W x D and concatenate with other layers we have extracted in this way, ending up at a H x W x Z descriptor.

For this project, you will not train the main classification network, in order to keep the required computation within reason on CPUs (though you are welcome to try if you have access to a GPU).  Instead, you will train a shallow CNN classifier on top of the extracted hypercolumns.  For this auxiliary classifier, you may experiment with different architectures, but general advice is to keep it below 10 layers (and maybe below 4 if training is too slow).  This classifier should predict a softmax distribution over pixels, i.e. a H x W x C tensor where C is the number of classes.

## Sparse Sampling and Fine-Tuning for Training

To make this project trainable on the CPU, we will need to initialize with a pre-trained network.  We will use the following sampling procedure, creating a training set of hypercolumn (also referred to as zoomout) features.  Once you have your hypercolumn/zoomout feature extractor working:
1. For each training image, sample 3 pixels from each class
2. For each pixel, save the pixel feature (i.e. 1x1xZ) dimension along with the identity of that pixel as a label
3. There are an average of 2.5 classes per image so this process should yield around 10,000 example (feature,label) pairs

Compute the mean and the variance of these Z dimensional features, and include the following in your network definition as the first layer: x = (x - self.mean)/self.std.

Next, train a shallow (3-layer) fully connected classifier on those features -- you will be classifying these Z-dimensional vectors into one of 21 classes.  See train_cls.py.

When that network is trained for a few epochs, modify the FC layers into 1x1 conv layers -- look at the torch.view function to see how to reshape d_in x d_out to d_out x d_in x 1 x 1.

Next, fine-tune the classifier on the actual training images.  See train_seg.py.

## Recommendations

1. When assembling hypercolumns, you may choose which layers of the main network (e.g. VGG11 or VGG16) to include in the hypercolumn feature.  A reasonable default is to include only the layers directly preceding a pooling layer.  This would result in Z=1472 in the case of VGG11.
2. Working with features at 112x112 spatial resolution (half of the input image resolution) is reasonable.

## Homework sections
----------------------------------
## PART I: Model Setup (30 points total)
   (1) Implement the 2D cross entropy function loss function in /losses/loss.py

   (2) Implement the hypercolumn (zoomout) feature extractor.

   (3) Implement a shallow neural network on top of the hypercolumn (zoomout) features to predict the most likely label (from the set of 21 PASCAL VOC classes) for each pixel.

----------------------------------
## PART II: Training (20 points total)
   (1) Sample and store a set of hypercolumn (zoomout) features as described in the sparse sampling procedure above.

   (2) Train your model.  You may wish to experiment with a variety of optimizers and learning rate schedules.

----------------------------------
## PART III: Testing (20 points total)
   (1) Evaluate your model on the PASCAL VOC validation set images (using the provided scripts).

   (2) Include example model output on several validation images.

----------------------------------
## Submit:
   (1) Source code (modified .py files and any additional files you wrote)

   (2) A write-up (hw4.pdf) briefly describing details of your architecture and training procedure, model accuracy, and example output on validation images
