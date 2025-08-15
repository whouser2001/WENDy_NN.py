"""
Torch implementation of WENDy-OLS
Only adequate for noiseless data and
linear-in-parameters ODE
"""
import torch

def WENDy_OLS(U, f, Phi, dPhi):
    """
    WENDy_OLS for noiseless data and
    linear-in-parameters ODE.

    Inputs:
    U : torch.tensor
        (M+1) x d tensor of noiseless data
    f : iterable
        size J vector of functions
        fj : R^(1 x d) -> R
    Phi : torch.tensor
        K x (M+1) tensor of test fn evals
    dPhi : torch.tensor
        K x (M+1) tensor of test fn derivative evals

    Output:
    W : torch.tensor
        J x D tensor of parameter estimates
    """
    M = U.shape[0] - 1
    J = len(f)

    Theta = [
        [f[j](U[m,:]) for j in range(J)]
        for m in range(M+1)
    ]
    Theta = torch.tensor(Theta)

    G = torch.matmul(Phi, Theta)
    B = -torch.matmul(dPhi, U)

    return torch.linalg.lstsq(G, B)

#test
if __name__ == '__main__':
    D = 3
    M = 4
    J = 2
    K = 1

    U = torch.ones((M+1, D))
    Phi = torch.ones((1, M+1))
    dPhi = 0.1*torch.ones((1, M+1))
    f = [
        lambda x : torch.sum(x),
        lambda x : torch.mean(x)
    ]

    print(WENDy_OLS(U,f,Phi,dPhi))
