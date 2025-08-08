"""
ODE Example problems
"""
import torch
import examples_class

import numpy as np
import matplotlib.pyplot as plt


def f_logistic(p,u,t): 
    """
    RHS for logistic
    u in R, p in R2
    """
    return torch.tensor(p[0]*u - p[1]*u**2)

def f_hindmarsh_rose(p,u,t):
    """
    RHS for Hindmarsh-Rose
    u in R3, p in R10
    """
    return torch.tensor([
        p[0]*u[1] - p[1]*u[0]**3 + p[2]*u[0]**2 - p[3]*u[2],
        p[4] - p[5]*u[0]**2 + p[6]*u[1],
        p[7]*u[0] + p[8] - p[9]*u[2]
    ])

def f_lorenz(p,u,t):
    """
    RHS for Lorenz oscillator
    u in R3, p in R3
    """
    return torch.tensor([
            p[0]*(u[1]-u[0]),
            u[0]*(p[1]-u[2]) - u[1],
            u[0]*u[1] - p[2]*u[2]
        ])

def f_goodwin_2d(p,u,t):
    """
    RHS for Goodwin (2D)
    u in R2, p in R5
    """
    return torch.tensor([
        p[0]/(36 + p[1]*u[1]) - p[2],
        p[3]*u[0] - p[4]
    ])

def f_goodwin_3d(p,u,t):
    """
    RHS for Goodwin (3D)
    u in R3, p in R7
    """
    return torch.tensor([
        p[0]/(2.15 + p[2]*u[2]**p[3]) - p[1]*u[0],
        p[4]*u[0] - p[5]*u[1],
        p[6]*u[1] - p[7]*u[2]
    ])

def f_sir_tdi(p,u,t):
    """
    RHS for SIR-TDI
    u in R3, in R5
    """
    e1 = torch.exp(-p[0]*p[1])
    C1 = u[2]*e1/(1-e1)
    C2 = p[3]*(1 - torch.exp(-p[4]*t**2))*u[1]
    return torch.tensor([
        -p[0]*u[0] + p[2]*u[1] + C1,
        p[0]*u[0] - p[2]*u[1] - C2,
        C2 - C1
    ])

#test
if __name__ == '__main__':
    
    generators = [
        examples_class.TorchODEDataGenerator(
            f = f_hindmarsh_rose,
            pstar = [10,10,30,10,10,50,
                     10,0.04,0.0319,0.01],
            u0 = [-1.31,-7.6,-0.2],
            T = 10
        ),
        examples_class.TorchODEDataGenerator(
            f = f_lorenz,
            pstar = [10,28,8/3],
            u0 = [2,1,1],
            T = 10
        )
    ]

    for G in generators:

        M = 25
        nr = 0.05
        Ustar = G.gen_Ustar(M).numpy()
        U = G.add_noise(Ustar,nr).numpy()
        #U = G.gen_U_noise(M,nr).numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(Ustar[:,0],Ustar[:,1],Ustar[:,2],label='truth',
                color='red')
        ax.scatter(U[:,0],U[:,1],U[:,2],label='noisy')
        ax.legend()

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    plt.show()