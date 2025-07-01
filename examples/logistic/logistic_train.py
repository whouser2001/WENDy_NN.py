"""
Train logistic model
"""
import os, sys
import logistic_data
sys.path.insert(0, '../../nn')
import nntools
import nnmodels

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


if __name__=='__main__':
    
    U = logistic_data.genU_logistic_noise(128, 0.1)

    logistic_testCNN = nntools.SupervisedLearningModel(
        model_name = 'logistic_testCNN',
        nn_module = nnmodels.testCNN,
        dataset = U,
        T = 10,
        gen_ODE = logistic_data.genU_logistic
    )

    logistic_testCNN.train_ensemble(
        nensemble = 2,
        nepochs = 5
    )



