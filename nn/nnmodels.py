"""
Store various architectures for ODE parameter learning
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class testCNN(nn.Module):
    """
    test NN on logistic input with only one conv layer/activation
    and one fully connected layer
    """
    def __init__(self):
        super().__init__()
        self.convLayer = nn.Conv1d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=3, 
            padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(16,2)
        self.double()

    def forward(self, x):

        x = self.relu(self.convLayer(x))
        x = self.pool(x)
        x = x.flatten()
        x = self.linear(x)

        return x

class logisticGrowthCNN(nn.Module):
    """
    LeNet for learning logistic growth parameters
    
    Input signal size is (M + 1) x 1 for M = 2**8, 2**9, 2**10
    So 8-10 conv layers? Extra conv layers for higher M?

    Output has size (2 x 1).
    """
    def init(self,
             p,
             set_dropout = False):
        
        super().__init__()

        if set_dropout: self.dropout = nn.Dropout(p=0.2)
        self.set_dropout = set_dropout

        conv1d = nn.Conv1d
        linear = nn.Linear 
        self.pool2 = nn.MaxPool1d(2,2)
        self.relu = nn.ReLU #max(0,x)

        D = 1
        nchannels = [D,5,5,10,10,25,25,100,100]

        #TODO
    
    def forward(self, x): raise NotImplementedError

#test
if __name__=='__main__':
    model = testCNN().to()
    #print(model)

    X = torch.rand(1,256)
    Y = model(X)
    print(Y)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")