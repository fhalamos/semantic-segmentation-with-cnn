import torch
import torch.nn.functional as F

def cross_entropy2d(predicts, targets):



  #->Not tested
  #->Not considering weighting classes for better accuracy
  
  loss = torch.nn.CrossEntropyLoss()
  output = loss(predicts, targets)

  return output


def cross_entropy1d(predicts, targets):
#->Whats the difference with 2d?
  return cross_entropy2d(predicts, targets)