# Standard Imports
import numpy as np
import copy
from time import time

# Jets
from pyjet import cluster

# ML
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

# SHAPER
from src.Manifolds import Manifold, Simplex


class Observable(nn.Module):

    def __init__(self, param_dict, sampler, beta, R, initializer=None, plotter=None) -> None:

        super().__init__()
        self.params = nn.ModuleDict(param_dict)
        self.sampler = sampler
        self.plotter = plotter
        self.keywords = []

        self.beta = beta
        self.R = R

        if initializer is None:
            self.initializer = param_dict.copy()
            self.initializer = None
        else:
            self.initializer = initializer

        self.freeze_list = {}
        self.is_trivial = (len(list(self.parameters())) == 0)

    # TODO: Implement "Simplify", parser when parameters have same names

    def __add__(self, observable):

        if isinstance(observable, Observable):

            # Get simplex weight

            new_param_dict = {}
            for obs in self.params:
                new_param_dict[obs] = copy.deepcopy(self.params[obs])
            for obs in observable.params:
                new_param_dict[obs] = copy.deepcopy(observable.params[obs])

            new_param_dict["Joint Weights"] = Simplex(2)

            # TODO: Allocate N better
            def new_sampler(N, new_param_dict):
                e1, z1 = self.sampler(N, new_param_dict)
                e2, z2 = observable.sampler(N, new_param_dict)

                # Concatenate and reweight
                e = torch.cat([e1, e2], dim=0)
                z = torch.cat([z1 * new_param_dict["Joint Weights"].params[0], z2 * new_param_dict["Joint Weights"].params[1]], dim=0)
                return e, z

            # def new_initializer

            def new_plotter(ax, new_param_dict):
                if self.plotter:
                    self.plotter(ax, new_param_dict)
                if observable.plotter:
                    observable.plotter(ax, new_param_dict)

            return Observable(new_param_dict, new_sampler, beta=self.beta, R=self.R, initializer=self.initializer, plotter=new_plotter)

            # return Manifold(self.parameters + manifold.parameters, self.labels + manifold.labels, [self, manifold])
        else:
            raise TypeError("Can only add Observables with other Observables, found %s" % type(observable))

    def sample(self, N):
        points, zs = self.sampler(N, self.params)
        return points, zs

    def enforce(self):

        for manifold in self.params:
            if self.params[manifold].params.requires_grad:
                self.params[manifold].enforce()

    def freeze(self, parameter, value=None):

        if value is not None:
            self.params[parameter].set(value)
        self.params[parameter].params.requires_grad = False

        self.freeze_list[parameter] = value

    def initialize(self, event):

        # Set default backup value if no initialization scheme exists
        for param in self.params:
            self.params[param].set(self.params[param].default_value)
            self.params[param].params.requires_grad = True

        # Initialization dictionary
        if (type(self.initializer) is dict):
            self.params = nn.ParameterDict(self.initializer.copy())

        # Default kt-initializer
        elif self.initializer == "kt":

            # Get the kt-clustering sequence
            N = self.params["Points"].shape[0]
            cluster_sequence = kt_initializer(event, self.R)
            jets = cluster_sequence.exclusive_jets(N)

            if "Points" in self.params.keys():

                # If N is too large, set the initializtion to the points themselves
                if (N >= event[0].shape[0]):
                    temp_points = np.zeros(self.params["Points"].shape)
                    temp_energies = np.zeros((self.params["Points"].shape[0],))

                    temp_points[:event[0].shape[0]] = event[0]
                    temp_energies[:event[0].shape[0]] = event[1]

                    points = temp_points
                    e = temp_energies

                else:
                    points, e = exclusive_jets(cluster_sequence, min(N, event[0].shape[0]))

                self.params["Points"].set(points)

                if "Weights" in self.params.keys():

                    num_weights = self.params["Weights"].N
                    energies = np.zeros((num_weights,))

                    if num_weights == N:
                        energies = e
                    elif num_weights == 2*N:
                        for i in range(N):
                            energies[i] = e[i]
                            # energies[i + N] = e[i] / 2

                    self.params["Weights"].set(energies / np.sum(energies))

                    # Freeze points, as in Apollonius
                    if N >= event[0].shape[0]:
                        self.freeze("Points")

                # Circle initializer based on clustering history
                if "Radius" in self.params.keys():

                    # if N > 1:
                    #     raise NotImplementedError("N > 1 circle initializer not implemented yet.")

                    reclustered = cluster(jets[0].constituents_array(), R=self.R, p=1)

                    # Make the assumption that the harder jet is the point and the softer one is the radius
                    p, e = exclusive_jets(cluster_sequence, N)
                    e = np.array(e)
                    num_weights = self.params["Weights"].N

                    r_ij = cdist(p, p)
                    radii = np.zeros((N,))
                    energies = np.zeros((num_weights,))
                    if num_weights == N:
                        energies = e
                    elif num_weights == 2*N:
                        for i in range(N):
                            energies[i] = e[i]

                        print(energies)
                        # energies[i + N] = e[i] / 2
                        # energies = np.ones((num_weights,)) / np.sum(e) / num_weights
                        # points = p[:N]

                        # for iter in range(N):

                        #     r_ij[r_ij == 0] = np.inf
                        #     d_ij = np.copy(r_ij) #* np.minimum(e[:, None], e[None, :])

                        #     closest_ij = np.unravel_index(d_ij.argmin(), d_ij.shape)
                        #     radii[iter] = r_ij[closest_ij]
                        #     if num_weights == N:
                        #         energies[iter] = e[closest_ij[0]] + e[closest_ij[1]]

                        #     # Choose the harder point to be the center
                        #     if e[closest_ij[0]] > e[closest_ij[1]]:
                        #         points[iter] = p[closest_ij[0]]
                        #         if num_weights == N*2:
                        #             energies[iter] = e[closest_ij[0]]
                        #             energies[N + iter] = e[closest_ij[1]]

                        #     else:
                        #         points[iter] = p[closest_ij[1]]
                        #         if num_weights == N*2:
                        #             energies[iter] = e[closest_ij[1]]
                        #             energies[N + iter] = e[closest_ij[0]]

                        #     # Next interation
                        #     mask = np.ones((2*(N - iter),), bool)
                        #     mask[closest_ij[0]] = False
                        #     mask[closest_ij[1]] = False
                        #     p = p[mask]
                        #     e = e[mask]
                        #     r_ij = cdist(p, p)

                    self.params["Radius"].set(radii)
                    self.params["Weights"].set(energies / np.sum(energies))
                    self.params["Points"].set(points)

            else:
                raise KeyError("Does not make sense to use kt to initialize a structure without points!")

        # Make sure things that should be frozen stay frozen
        for param in self.freeze_list:
            self.freeze(param, self.freeze_list[param])

    def draw(self, ax):
        if self.plotter is None:
            pass
        else:
            self.plotter(ax, self.params)

    def get_dict(self, dev=None):

        if dev is None:
            dev = self.device()

        dictionary = {}
        for manifold in self.params:

            if dev == torch.device("cpu"):
                dictionary[manifold] = self.params[manifold].params.clone().detach().numpy()
            else:
                dictionary[manifold] = self.params[manifold].params.clone().cpu().detach().numpy()

        return dictionary


def kt_initializer(event, R):

    y, z = event

    four_vectors = []
    for (y_i, z_i) in zip(y, z):
        v = (z_i, y_i[0], y_i[1], 0)
        four_vectors.append(v)
    four_vectors = np.array(four_vectors, dtype=[("pt", "f8"), ("eta", "f8"), ("phi", "f8"), ("mass", "f8")])
    sequence = cluster(four_vectors, R=R, p=1)
    return sequence


def exclusive_jets(sequence, N):

    jets = sequence.exclusive_jets(N)

    # Apply initialization
    jets = jets[:N]
    initialization = []
    energies = []
    for jet in jets:
        initialization.append([jet.eta, jet.phi])
        energies.append(jet.pt)
    initialization = np.array(initialization).astype(np.float32)
    return initialization, energies
