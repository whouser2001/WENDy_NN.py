"""
class to generate data for ODE examples
"""
import torch
from torchdiffeq import odeint

class TorchODEDataGenerator():

    def __init__(self, 
                 f=None, 
                 pstar=None, 
                 u0=None, 
                 T=-1):
        """
        Class that provides utilities for generating
        datasets U, Ustar in Pytorch for a specified
        ODE example problem

        Parameters
        ----------
        f: ODE RHS with parameters
            p : iterable
                model parameters
            u : iterable
                point in space
            t : int
                corresponding time scalar

        pstar : iterable
            True ODE parameters
        u0 : iterable
            True u0 of the problem
        T : int
            Defines time range [0,T]
        """
        self.f = f
        self.fstar = lambda t,u : f(pstar,u,t)
        self.pstar = torch.Tensor(pstar)
        self.u0 = torch.Tensor(u0)
        self.T = T
    
    def solve_ODE(self,p,t,u0): 
        """
        Torch ODE solve for parameters p at times t
        given initial data u0
        """
        return odeint(lambda t,u: self.f(p,u,t),u0,t)

    def gen_parametrized_U(self,p,u0,M):
        """
        Generate M+1 noiseless data points with parameters
        p and initial data u0
        """
        t = torch.linspace(0,self.T,M+1)
        return self.solve_ODE(p,t,u0)
    
    def gen_Ustar(self,M):
        """
        Generate true model data
        """
        return self.gen_parametrized_U(self.pstar,self.u0,M)
    
    def gen_U_noise(self,M,nr):
        """
        Generate data U = U* + E with Gaussian noise
        TODO: Extend to arbitrary noise scheme
        """
        Ustar = self.gen_Ustar(M)
        E = nr*torch.normal(
            mean=torch.zeros(Ustar.shape),
            std=torch.ones(Ustar.shape)
        )
        return Ustar + E

    def add_noise(self,Ustar,nr):
        """
        Add Gaussian noise to data
        TODO: Extend to arbitrary noise scheme
        """
        E = nr*torch.normal(
            mean=torch.zeros(Ustar.shape),
            std=torch.ones(Ustar.shape)
        )
        return Ustar + E