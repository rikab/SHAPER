# Standard Imports
import numpy as np
from time import time
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')

# Jets
from pyjet import cluster
from src.Observables import Observable

# ML
import torch
import torch.nn as nn

# SHAPER 
from src.Manifolds import Manifold, Simplex, Coordinates2D


# Sample from a normalized uniform distribution
def uniform_sampler(N, param_dict):
    points = torch.FloatTensor(N, 2).uniform_(-R, R)
    zs = torch.ones((N,)) / N
    return (points, zs)





class NSubjettiness(Observable):

    def __init__(self, N, beta, R, initializer="kt") -> None:

        self.N = N

        # Sample at N weighted Dirac deltas
        def point_sampler(N, param_dict):
            return (param_dict["Points"].params, param_dict["Weights"].params)

        def plotter(ax, param_dict):
            points = param_dict["Points"].params.clone().detach().numpy()
            weights = param_dict["Weights"].params.clone().detach().numpy()
            ax.scatter(points[:,0], points[:,1], color = "Purple", label = "Subjets", s = 2 * weights * 500/np.sum(weights), alpha = 0.5)

            for i in range(N):
                plt.text(0.05, 0.06 + 0.03*i, "Point: (%.3f, %.3f), Weight: %.3f" % (points[i,0], points[i,1], weights[i]), fontsize = 10, transform = plt.gca().transAxes)

        super().__init__({"Points" : Coordinates2D(self.N), "Weights" : Simplex(self.N)}, point_sampler, beta, R, initializer, plotter)




