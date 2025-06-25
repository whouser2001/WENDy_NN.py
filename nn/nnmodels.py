"""
Store various architectures for ODE parameter learning
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class logisticGrowthCNN(nn.Module):
    """
    CNN for learning logistic growth parameters
    
    Input signal size is (M + 1) x 1 for M = 2**8, 2**9, 2**10
    So 8-10 layers? Extra conv layers for higher M?

    Output has size (2 x 1)
    """