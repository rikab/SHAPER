# Standard Imports
import numpy as np
from time import time

# Jets
from pyjet import cluster

# ML
import torch
import torch.nn as nn

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

            new_param_dict = {**self.params, **observable.params}
            new_param_dict["Joint Weights"] = Simplex(2)

            # TODO: Allocate N better
            def new_sampler(N, new_param_dict):
                e1, z1 = self.sampler(N, new_param_dict)
                e2, z2 = observable.sampler(N, new_param_dict)

                # Concatenate and reweight
                e = torch.cat([e1, e2], dim=0)
                z = torch.cat([z1 * new_param_dict["Joint Weights"].params[0], z2 * new_param_dict["Joint Weights"].params[1]], dim=0)
                return e, z

            def new_plotter(ax, new_param_dict):
                self.draw(ax)
                observable.draw(ax)

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
                    energies = temp_energies

                else:
                    points, energies = exclusive_jets(cluster_sequence, min(N, event[0].shape[0]))

                self.params["Points"].set(points)

                if "Weights" in self.params.keys():

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
                    p, e = exclusive_jets(reclustered, N*2)
                    r = np.sqrt(np.sum(np.square(p[0] - p[1])))

                    self.params["Radius"].set(torch.ones((N,)) * r)
                    # self.params["Radius"].set(torch.ones((1,)) * self.R / 2)
                    num_weights = self.params["Structure Weights"].N
                    p, e = exclusive_jets(reclustered, num_weights)
                    self.params["Structure Weights"].set(e / np.sum(e))

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
