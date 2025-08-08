import os, sys
import logistic_data
sys.path.insert(0, '../../nn')
import nntools
import nnmodels

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    
    pstar = np.ones(2)
    nnr = 50
    nr = np.linspace(1,50,nnr)*0.01 #noise ratios
    Ms = [256] #,512,1028 later
    epochs = 1

    CREs = np.ones(nnr)*(-1)
    legs = ['M=256','M=512','M=1028']

    for i in range(len(Ms)):
        M = Ms[i]

        for j in range(nnr):
            r = nr[j]

            U = logistic_data.genU_logistic_noise(M, r)

            logistic_testCNN = nntools.ODESupervisedLearningModel(
                model_name = 'logistic_testCNN',
                nn_module = nnmodels.testCNN,
                dataset = U,
                T = 10,
                gen_ODE = logistic_data.genU_logistic
            )

            try: #an ODE solve could fail and throw error
                models = logistic_testCNN.train_ensemble(
                    nensemble = 1,
                    nepochs = epochs,
                    learn_rate=5e-3,
                    tol = 1e-4
                )

                p_hat = logistic_testCNN.predict_dataset().detach()
                p_hat = np.array(p_hat)
                CRE = np.linalg.norm(p_hat-pstar, ord=2)/np.sqrt(2)
                CREs[j] = CRE

            except:
                print('ODE solve fail. M = {}, nr = {}'.format(
                    M, r
                ))

        CREs = np.where(CREs < 0, np.nan, CREs)
        
        plt.plot(nr, CREs, label=legs[i])
        plt.legend()
    plt.show()
