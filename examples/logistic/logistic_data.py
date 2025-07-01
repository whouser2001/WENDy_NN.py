"""
logistic equation
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f(p,u,t): return p[0]*u - p[1]*u**2

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

def genU_logistic(p, Tsubset):
    """
    For setting up loss in the NN.

    p : parameters [p1, p2]
    Tsubset : subset of t after masking for
        train/validation datasets
    """
    u = solve_logistic(p,10,0.01)
    return np.array([u(ti) for ti in Tsubset])
