import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
import shutil
import os, sys


def load_dataset(filename, N, mass_lower = 0, mass_upper = np.inf, eta_cut = 1.3, pt_cut = 300):


        n = 0
        i = 0
        X = np.load(filename, "r",)[:N*100]
        X = np.copy(X)
        print(X.shape)
    
        # Normalize & Center
        events = []
        masses = []
        while n < N:
        # for x in X:
            x = X[i]
            pt = np.sum(x[:,0])
            zs = x[:,0] / np.sum(x[:,0])
            
            mask = zs[:] > 0
            x[mask, 2] = np.unwrap(x[mask,2])
            yphi_avg = np.average(x[mask,1:3], weights = x[mask,0], axis = 0)
            x[mask,1:3] -= yphi_avg  

            px = np.sum(x[:,0] * np.cos(x[:,2]))
            py = np.sum(x[:,0] * np.sin(x[:,2]))
            pz = np.sum(x[:,0]*np.sinh(x[:,1]))
            e = np.sum(x[:,0] * np.cosh(x[:,1]))

            mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
            masses.append(mass)

            print(mass)
            if mass_lower < mass < mass_upper and np.abs(yphi_avg[0]) < eta_cut and pt > pt_cut:
                events.append( (x[:,1:3] , zs ) )
                n += 1
            i += 1
        return events, masses