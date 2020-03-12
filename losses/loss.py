import torch
import torch.nn.functional as F
import torch.nn as nn


USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def cross_entropy2d(predicts, targets):


  #->Not considering weighting classes for better accuracy
 

  unique_targets, counts = torch.unique(targets, return_counts=True)
 
  logsoftmax = nn.LogSoftmax(dim=1)

  #weight (Tensor, optional) – a manual rescaling weight given to each class. If given, it has to be a Tensor of size C. Otherwise, it is treated as if having all ones.
  weights = torch.zeros([21], dtype=torch.float64)
  for i in range(len(unique_targets)):
    weights[unique_targets[i]] = 1/counts[i].float()

  weights = weights.to(device=device)

  loss = nn.NLLLoss(weight=weights.float())

  output = loss(logsoftmax(predicts),targets)  

  return output


def cross_entropy1d(predicts, targets):

  logsoftmax = nn.LogSoftmax(dim=1)

  loss = nn.NLLLoss()

  output = loss(logsoftmax(predicts),targets)  

  return output