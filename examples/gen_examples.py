"""
ODE Example problems used in the WENDy-MLE paper
"""
import torch
import ODE_data

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

params = {
    'logistic' : {
        'f' : f_logistic,
        'pstar' : [1,-1],
        'u0' : [0.01],
        'T' : 10
    },
    'hindmarsh_rose' : {
        'f' : f_hindmarsh_rose,
        'pstar' : [10,10,30,10,10,50,
                    10,0.04,0.0319,0.01],
        'u0' : [-1.31,-7.6,-0.2],
        'T' : 10
    }, 
    'lorenz' : {
        'f' : f_lorenz,
        'pstar' : [10,28,8/3],
        'u0' : [2,1,1],
        'T' : 10
    },
    'goodwin_2d' : {
        'f' : f_goodwin_2d,
        'pstar' : [72,1,2,1,1],
        'u0' : [7,-10],
        'T' : 60
    },
    'goodwin_3d' : {
        'f' : f_goodwin_3d,
        'pstar' : [3.4884,0.0969,10,0.0969,
                0.0581,0.0969,0.0775],
        'ustar' : [0.3617,0.9137,1.3934],
        'T' : 80
    },
    'sir-tdi' : {
        'f' : f_sir_tdi,
        'pstar' : [1.99,1.5,0.074,0.113
                ,0.0024],
        'u0' : [1,0,0],
        'T' : 50
    }}

#test
if __name__ == '__main__':

    examples = ['logistic', 'hindmarsh_rose',
                'lorenz', 'goodwin_2d',
                'goodwin_3d', 'sir-tdi']

    for name in examples:

        G = ODE_data.TorchODEDataGenerator(
            **params[name]
        )

        M = 50
        nr = 0.05
        Ustar = G.gen_Ustar(M)

        print(name)
        print(Ustar)
        print('\n')
        #U = G.add_noise(Ustar,nr).numpy()
        #Ustar = Ustar.numpy()
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111,projection='3d')
        #ax.plot(Ustar[:,0],Ustar[:,1],Ustar[:,2],label='truth',
        #        color='red')
        #ax.scatter(U[:,0],U[:,1],U[:,2],label='noisy')
        #ax.legend()

        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('z')
    
    #plt.show()