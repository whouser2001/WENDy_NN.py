import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SupervisedLearningModel():

    def __init__(self,
                 model_name='',
                 nn_module=None):
        """
        Class that provides utilities used for training NN model

        Parameters
        ----------
        model_name : str

        nn_module : torch.nn.Module
        Pytorch NN module to be used for trainging (from nnmodels)

        hyperHyperParameters : list
        For now, additional shape parameters to add/subtract layers
        to account for inputs of varying sizes. Could be extended
        to contain other information.
        """
        self.model_name = model_name
        self.nn_module = nn_module

        def train_ensemble(self,
                           nensemble = 10,
                           torch_loss = nn.MSELoss,
                           torch_optimizer = optim.Adam,
                           stop = 'nepochs',
                           weight_decay = 0.0,
                           learn_rate = 1e-4,
                           nepochs = 100,
                           tol = 1e-1): raise NotImplementedError

        def compute_loss_batch(self): raise NotImplementedError

        def predict_dataset(self): raise NotImplementedError

    