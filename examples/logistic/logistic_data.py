"""
logistic equation
"""
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torchdiffeq import odeint #https://github.com/rtqichen/torchdiffeq?tab=readme-ov-file
import matplotlib.pyplot as plt

def f(p,u,t): 
    return p[0]*u - p[1]*u**2

def solve_logistic(p, T, u0):

    def fhat(t,u): return f(p,u,t)
    solve_data = solve_ivp(
        fhat, (0,T), u0, dense_output=True
    )
    u = solve_data['sol']
    return u

def genU_logistic_noise(M, nRat):
    """
    generate data U = U* + E as in paper
    """
    pstar = [1,1]
    T = 10
    u0 = [0.01]
    u = solve_logistic(pstar, T, u0) #true u

    t = np.linspace(0,T,M+1)
    Ustar = np.array([u(t[i]) for i in range(len(t))])
    E = nRat*np.random.normal(size=(Ustar.size,1))

    return Ustar + E

def solve_logistic_torch(p, T, u0):
    """
    Assumes p,T are PyTorch tensors. Necessary to
    preserve the gradient while computing loss

    Uses a torch-based solver which avoids numpy
    operations (which kill the gradient)
    """
    def fhat(t,u): return f(p,u,t)

    u = odeint(fhat, T, torch.tensor(u0))
    return u.flatten()


def genU_logistic(p, T, mask):
    """
    For setting up loss in the NN.

    p : parameters [p1, p2] as torch tensor
    T : full time span as torch tensor
    Tsubset : subset of t after masking for
        train/validation datasets
    """
    u = solve_logistic_torch(p,T,[0.01])

    mask = torch.tensor(mask, dtype=bool)
    u = u[mask]
    
    return u
