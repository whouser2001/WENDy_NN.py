import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SupervisedLearningModel():

    def __init__(self,
                 model_name='',
                 nn_module=None,
                 hyperHyperParameters=None):
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
        self.hhParameters = hyperHyperParameters

    