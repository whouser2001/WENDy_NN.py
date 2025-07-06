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
    
    U = logistic_data.genU_logistic_noise(256, 0.01)

    logistic_testCNN = nntools.ODESupervisedLearningModel(
        model_name = 'logistic_testCNN',
        nn_module = nnmodels.testCNN,
        dataset = U,
        T = 10,
        gen_ODE = logistic_data.genU_logistic
    )

    models = logistic_testCNN.train_ensemble(
        nensemble = 2,
        nepochs = 20,
        learn_rate=5e-3
    )

    print(models[1])



