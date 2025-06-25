"""
logistic equation
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f(p,u,t): return p[0]*u - p[1]*u**2

pstar = [1,1]
J = len(pstar)
def fstar(t,u): return f(pstar,u,t)
u0 = [0.01]
T = 10
solve_data = solve_ivp(fstar, (0,T), u0, dense_output=True)
u = solve_data['sol']

def genU_logistic(M, nRat):

    t = np.linspace(0,T,M+1)
    Ustar = np.array([u(t[i]) for i in range(len(t))])
    E = nRat*np.random.normal(size=(Ustar.size,1))

    return Ustar + E


if __name__ == '__main__':
    t = np.linspace(0,T,100)
    plt.plot(t, np.squeeze(u(t)))
    U = genU_logistic(99, 0.005)
    #print(U.shape)
    plt.plot(t, U)
    plt.show()