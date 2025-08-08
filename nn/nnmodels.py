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

class WENDyTestFunctionSimpleCNN(nn.Module):
    """
    Apply an unbiased convolution layer no bias to WENDy input data
    """
    def __init__(self):
        super().__init__()

        #Parameters determining kernel size
        M = 1024 #num data points
        r = 1/3 #test function radius
        T = 10 #gives time range [0,T]

        K = 1 + 2*np.floor(r*self.M/T)

        self.M = M
        self.convLayerF = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(K,1),
            bias=False
        )
        self.convLayerU = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(K,1),
            bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        
        M = self.M
        F,U = torch.split(X, M+1, 0)
        Q = torch.eye(M+1)

        F = torch.matmul(Q,F)
        U = torch.matmul(Q,U)

        WF = self.convLayerF(F)
        WU = self.convLayerU(U)

        Y = torch.cat(WF,WU)

        return Y

#test
if __name__=='__main__':
    model = testCNN().to()
    #print(model)

    X = torch.rand(1,256)
    Y = model(X)
    print(Y)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")