import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ODESupervisedLearningModel():

    def __init__(self,
                 model_name='',
                 nn_module=None,
                 dataset=None,
                 batch_size=1,
                 train_device='cpu',
                 test_device='cpu'):
        """
        Class that provides utilities used for training NN model

        Parameters
        ----------
        model_name : str

        nn_module : torch.nn.Module
        Pytorch NN module to be used for trainging (from nnmodels)
        """
        self.model_name = model_name
        self.nn_module = nn_module
        self.dataset = dataset
        self.batch_size = batch_size

        self.info_dict = {}
        self.info_dict['model name'] = self.model_name
        self.info_dict['batch_size'] = batch_size
        self.info_dict['train_device'] = train_device
        self.info_dict['test_device'] = test_device

        def train_ensemble(self,
                           nensemble = 1,
                           torch_loss = nn.MSELoss,
                           torch_optimizer = optim.Adam,
                           stop = 'nepochs',
                           weight_decay = 0.0,
                           learn_rate = 1e-4,
                           nepochs = 100,
                           tol = 1e-1): 
                
                info_dict = self.info_dict
                model_name = info_dict['model_name']
                device = info_dict['train_device']

                # store training hyperparameters
                info_dict['nensemble'] = nensemble
                info_dict['nepochs'] = nepochs
                info_dict['torch_optimizer'] = torch_optimizer.__name__
                info_dict['torch_loss'] = torch_loss.__name__
                info_dict['learn_rate'] = learn_rate
                info_dict['weight_decy'] = weight_decay
                info_dict['tolerance'] = tol

                #TODO: implement step to divide data into batches
                compute_loss = self.compute_loss_batch
                
                #TODO: Random seed?
                #TODO: save_model_info?

                self.stop_epoch = np.zeros(nensemble, dtype=int) - 1
                self.stop = stop
                self.tol = tol

                # begin training
                for model_index in range(nensemble):
                     
                    # define model
                    nn_model = self.nn_module()
                    nn_model.to(device)

                    # train
                    loss_func = torch_loss()
                    optimizer = torch_optimizer(nn_model.parameters(),
                                                 lr=learn_rate,
                                                 weight_decay=weight_decay)
                     
                    #loss per epoch
                    train_loss_array = np.zeros(nepochs) - 1
                    valid_loss_array = np.zeros(nepochs) - 1

                    epoch = 1
                    while epoch < nepochs + 1:
                         
                        #train mode
                        nn_model.train()
                        nn_model.to(device)

                        #monitor training loss
                        train_loss = 0.0

                        #TODO: extend to multiple batches
                        optimizer.zero_grad()
                        loss = compute_loss(nn_model, loss_func, device=device)

                        #backpropogation
                        loss.backward()
                        optimizer.step()

                        train_loss = loss.item()
                        train_loss_array[epoch-1] = train_loss

                        #TODO: save_loss?
                        #TODO: validation? I don't think we need this

                        #display status
                        msg = "\nmodel {:2d}/{:2d}, e= {:3d}, tl= {:1.2e}, vl= {:1.2e}"\
                        .format(model_index, nensemble-1,
                                epoch, train_loss)
                        self._print(msg)

                        #apply stopping criteria
                        #TODO: implement _save_nn_model?
                        if train_loss < tol or epoch == nepochs:
                            
                            self.stop_epoch[model_index] = epoch
                            self.info_dict['stop_epoch_list'] = self.stop_epoch.tolist()
                            break

                        epoch += 1

        def compute_loss_batch(self,
                               nn_model, loss_func, 
                               batch_index=None, device='cpu',
                               train_test_ratio=0.7):
            """
            Get parameters, then generate data according to
            ODE and compare with remaining points
            """
            #TODO: Extend to multiple batches
            #TODO: train/test split
            data = self.dataset

            #select ttr% of the indices in data
            data.to(device)

            model_out = nn_model(data)

            raise NotImplementedError

        def predict_dataset(self): raise NotImplementedError

    