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
                 T=None,
                 gen_ODE=None,
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

        dataset : current np array (TODO: change to torch tensor)
            Sampled data U

        T : int
            Defines the time interval (0,T) over which we sampled data

        gen_ODE : Function with parameters
            p : torch.tensor
                ODE paramters
            t : torch.tensor
                Specifies times on which U is sampled
                This is itself parametrized by T, U.size
            mask : np.array (TODO?: change to torch tensor)
                Mask applied to U (for train/val split)

            Uses an ODE solve (torchdiffeq.odeint) to generate data
            according to p for comparison with U[mask] in loss function
        """
        self.model_name = model_name
        self.nn_module = nn_module
        self.dataset = dataset
        self.batch_size = batch_size
        self.gen_ODE = gen_ODE
        self.T = T

        self.info_dict = {}
        self.info_dict['model_name'] = self.model_name
        self.info_dict['batch_size'] = batch_size
        self.info_dict['train_device'] = train_device
        self.info_dict['test_device'] = test_device
        self.info_dict['shape'] = dataset.shape
        self.info_dict['size'] = dataset.size
        self.info_dict['T'] = T

    def train_ensemble(self,
                        nensemble = 1,
                        torch_loss = nn.MSELoss,
                        torch_optimizer = optim.Adam,
                        stop = 'nepochs',
                        weight_decay = 0.0,
                        learn_rate = 5e-3,
                        nepochs = 100,
                        tol = 1e-2): 
            
            info_dict = self.info_dict
            model_name = info_dict['model_name']
            device = info_dict['train_device']
            U = self.dataset

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

            self.models = []
            models = self.models

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

                #train/validation mask to be applied to t and U
                #TODO: make tts a hyperparameter
                train_test_split = 0.7
                n_elements = info_dict['size']
                n_tomask = int(train_test_split*n_elements)

                masked_indices = np.random.choice(n_elements,n_tomask,
                                                  replace=False)
                mask = np.zeros(n_elements, dtype=bool)
                mask[masked_indices] = True
                maskU = mask.reshape(info_dict['shape'])

                U_train = U.copy().astype(float)[maskU]
                U_train = np.expand_dims(U_train, axis=0)
                U_train = torch.tensor(U_train, dtype=torch.float64,
                                       requires_grad=True)

                U_val = U.copy().astype(float)[
                    np.logical_not(maskU)
                ]
                U_val = np.expand_dims(U_val, axis=0)
                U_val = torch.tensor(U_val, dtype=torch.float64,
                                     requires_grad=True)

                #Corresponding times of U
                t = np.linspace(0,self.T,n_elements)
                t = torch.tensor(t, dtype=torch.float64)

                epoch = 1
                while epoch < nepochs + 1:
                        
                    #train mode
                    nn_model.train()
                    nn_model.to(device)

                    #monitor training loss
                    train_loss = 0.0

                    optimizer.zero_grad()
                    loss = compute_loss(
                        U_train, t, mask, nn_model, 
                        loss_func, device=device)

                    #backpropogation
                    loss.backward()

                    optimizer.step()

                    train_loss = loss.item()
                    train_loss_array[epoch-1] = train_loss

                    #switch model to validation mode
                    nn_model.eval()

                    mask_not = np.logical_not(mask)
                    loss = compute_loss(
                        U_val, t, mask_not, nn_model,
                        loss_func, device=device
                    )

                    valid_loss = loss.item()
                    valid_loss_array[epoch-1] = valid_loss

                    #display status
                    gap = 5
                    if not epoch % gap:
                        msg = "\nmodel {:2d}/{:2d}, e= {:3d}, tl= {:1.2e}, vl= {:1.2e}"\
                        .format(model_index, nensemble-1,
                                epoch, train_loss, valid_loss)
                        print(msg)

                    #apply stopping criteria
                    #TODO: implement _save_nn_model?
                    if train_loss < tol or epoch == nepochs:
                        
                        self.stop_epoch[model_index] = epoch
                        self.info_dict['stop_epoch_list'] = self.stop_epoch.tolist()
                        models.append(nn_model)
                        
                        break

                    epoch += 1

            return self.models

    def compute_loss_batch(self, U_subset, time_span, mask,
                            nn_model, loss_func, batch_index=None, 
                            device='cpu'):
        """
        Get parameters, then generate data according to
        ODE and compare with remaining points
        """
        #TODO: Extend to multiple batches
        
        p_hat = nn_model(U_subset)
        p_hat = p_hat.squeeze()

        mask = torch.tensor(mask, dtype=torch.bool)
        U_hat = self.gen_ODE(p_hat, time_span, mask)
        #U_hat.to(torch.float32)
        
        U_subset = U_subset.flatten()
        loss = loss_func(U_hat, U_subset) 

        return loss

    def predict_dataset(self): 
        """
        Predict the parameters based on entire dataset U
        and all models in the ensemble
        """
        
        U = torch.tensor(self.dataset.T)

        p_hat = torch.zeros(2)
        models = self.models

        for model in models: p_hat += model(U)
        p_hat /= len(models)

        return p_hat

class WENDyTestFunctionSupervisedLearningModel():
    
    def __init__(self,
                 model_name='',
                 nn_module=None,
                 dataset=None,
                 WENDy_variant='OLS',
                 batch_size=1,
                 train_device='cpu',
                 test_device='cpu'):
        """
        Class that provides utilites for test function learning
        in the WENDy problem

        Parameters
        ----------
        model_name : str

        nn_module : torch.nn.Module
            Pytorch NN module to be used in training

        dataset : torch.tensor
            Tensor of sampled data U

        WENDy_variant : str
            specifies which WENDy variant to run for computing loss
        """
        self.model_name = model_name
        self.nn_module = nn_module
        self.dataset = dataset
        self.batch_size = batch_size
        self.WENDy_variant = WENDy_variant

        self.info_dict = {}
        self.info_dict['model_name'] = self.model_name
        self.info_dict['batch_size'] = batch_size
        self.info_dict['train_device'] = train_device
        self.info_dict['test_device'] = test_device
        self.info_dict['shape'] = dataset.shape
        self.info_dict['size'] = dataset.size
        
    def train_ensemble(self,
                       nensemble = 1,
                       torch_loss = nn.MSELoss,
                       torch_optimizer = optim.Adam,
                       stop = 'nepochs',
                       weight_decay = 0.0,
                       learn_rate = 5e-3,
                       nepochs = 100,
                       tol = 1e-3):
        
        info_dict = self.info_dict
        model_name = info_dict['model_name']
        device = info_dict['train_device']
        U = self.dataset

        info_dict['nensemble'] = nensemble
        info_dict['nepochs'] = nepochs
        info_dict['torch_optimizer'] = torch_optimizer.__name__
        info_dict['torch_loss'] = torch_loss.__name__
        info_dict['learn_rate'] = learn_rate
        info_dict['weight_decy'] = weight_decay
        info_dict['tolerance'] = tol

        compute_loss = self.compute_loss_batch

        #TODO: save_model_info?

        self.stop_epoch = np.zeros(nensemble, dtype=int) - 1
        self.stop = stop
        self.tol = tol

        self.models = []
        models = self.models

        #training below

    def compute_loss_batch(): 
        #Get dPhi*U, Phi*F
        # Or dPhi, Phi

        #Plug into WENDy thing
        raise NotImplementedError

    def predict_dataset(): raise NotImplementedError

    