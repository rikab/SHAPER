import torch
import torch.nn as nn
import numpy as np


class Manifold(nn.Module):

    def __init__(self):
        super().__init__()
        self.N = 0

    def set(self, params):

        if (self.params.shape == params.shape):
            with torch.no_grad():
                if isinstance(params, np.ndarray):
                    self.params.copy_(torch.Tensor(params))
                else:
                    self.params.copy_(params)
        else:
            raise ValueError("Expected shape " + str(self.params.shape) + ", got shape " + str(params.shape))


class Coordinates2D(Manifold):

    def __init__(self, N, R=1):
        super().__init__()
        self.N = N
        self.R = R
        self.params = torch.nn.Parameter(torch.FloatTensor(N, 2).uniform_(-R, R))
        self.default_value = torch.FloatTensor(N, 2).uniform_(-R, R)
        self.shape = self.params.shape

    def enforce(self):
        pass


class Simplex(Manifold):

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.params = torch.nn.Parameter(torch.ones((N,)) / N)
        self.default_value = torch.ones((N,)) / N

    def enforce(self):

        # pass
        cnt_n = torch.arange(self.N, device=self.params.device)
        u = self.params.sort(descending=True).values
        v = (u.cumsum(dim=0) - 1) / (cnt_n + 1)
        w = v[(u > v).sum() - 1]
        self.set((self.params - w).relu())


class PositiveReals(Manifold):

    def __init__(self, N, initializer=1):
        super().__init__()
        self.N = N
        self.params = torch.nn.Parameter(initializer * torch.ones((N,)) / N)
        self.default_value = initializer * torch.ones((N,)) / N

    def enforce(self):
        self.set(self.params.relu())


# lass Manifold():

#     def __init__(self, parameters, labels, submanifolds):

#         self.labels = labels
#         self.parameters = parameters
#         self.submanifolds = submanifolds

#     def __mul__(self, manifold):

#         if isinstance(manifold, Manifold):
#             return Manifold(self.parameters + manifold.parameters, self.labels + manifold.labels, [self, manifold])
#         else:
#             raise TypeError("Can only multiply Manifolds by other Manifolds, found %s" % type(manifold))

#     def simplify(self):
#         pass

#     def enforce(self):
#         pass


# class Singleton():


#     def __init__(self):
#         pass


# class Coordinates(Manifold):

#     def __init__(self, parameters, labels, submanifolds):
#         super().__init__(parameters, labels, submanifolds)


# class Simplex(Manifold):

#     def __init__(self, parameters, labels, submanifolds):
#         super().__init__(parameters, labels, submanifolds)

# class Reals():

#     def __init__(self, ) -> None:
#         pass
