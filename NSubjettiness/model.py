import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KDS(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        Rprime,
        R,
        graph,
        accelerate=True,
        train_step=True,
        W=None,
        step=None,
        weighted=None
    ):

        super(KDS, self).__init__()

        # hyperparameters
        self.register_buffer("num_layers", torch.tensor(int(num_layers)))
        self.register_buffer("input_size", torch.tensor(int(input_size)))
        self.register_buffer("hidden_size", torch.tensor(int(hidden_size)))  # Added +1 for unclustered anchor
        self.register_buffer("Rprime", torch.tensor(float(Rprime)))
        self.register_buffer("R", torch.tensor(float(R)))
        self.graph = graph
        self.register_buffer("accelerate", torch.tensor(bool(accelerate)))

        # parameters
        if W is None:
            W = torch.zeros(self.hidden_size, self.input_size)
        self.register_parameter("W", torch.nn.Parameter(W))
        if step is None:
            step = W.svd().S[0] ** -2
        if train_step:
            self.register_parameter("step", torch.nn.Parameter(step))
        else:
            self.register_buffer("step", step)

    def forward(self, y):
        x = self.encode(y)
        y = self.decode(x)
        return y

    def encode(self, y):
        if self.accelerate:
            return self.encode_accelerated(y)
        else:
            return self.encode_basic(y)

    def point_of_closest_approach(self, y, polygon):

        x_tmp = torch.zeros(y.shape[0], polygon.shape[0], device=y.device)
        x_old = torch.zeros(y.shape[0], polygon.shape[0], device=y.device)
        A = self.W[polygon]

        for layer in range(self.num_layers):

            grad = torch.zeros(y.shape[0], polygon.shape[0], device=y.device)
            Q = torch.nan_to_num(y - x_tmp @ A)       # Q has dim [points, dim]
            d = (Q.unsqueeze(1) * A.unsqueeze(0))   # d has dim [points, anchors, dim]

            grad = -1 * d.sum(dim=2)

            x_new = self.activate(x_tmp - grad * self.step)
            x_old, x_tmp = x_new, x_new + layer / (layer + 3) * (x_new - x_old)

        # print(A - (x_new @ A) )
        return x_new @ A

    def encode_basic(self, y):

        x = torch.zeros(y.shape[0], self.hidden_size+1, device=y.device)
        # weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
        weight = (
            y.square().sum(dim=1, keepdims=True)
            + self.W.T.square().sum(dim=0, keepdims=True)
            - 2 * y @ self.W.T
        )
        for layer in range(self.num_layers):
            grad = torch.zeros(y.shape[0], self.hidden_size + 1)
            # grad[:,1:] = (x @ self.W - y) @ self.W.T
            grad[:, 1:] = grad[:, 1:] + weight * self.penalty
            # grad[:,0:] = (x @ self.W - y) @ self.W.T
            grad[:, 0:] = grad[:, 0:] + self.R**2 * self.penalty
            x = self.activate(x - grad * self.step)
        return x[:, 1:], x[:, 0]

    def encode_accelerated(self, y):

        beta = 1.0
        num_polygons = len(self.graph.structure)

        f_tmp = torch.zeros(y.shape[0], num_polygons+1, device=y.device)
        f_old = torch.zeros(y.shape[0], num_polygons+1, device=y.device)
        f0 = f_tmp[:, 0]

        points = []
        for polygon in self.graph.structure:
            points.append(self.point_of_closest_approach(y, polygon))

        for layer in range(self.num_layers):

            grad = torch.zeros(y.shape[0], num_polygons+1, device=y.device)
            grad[:, 0] = 1

            for i, polygon in enumerate(self.graph.structure):
                grad[:, i+1] = ((y - points[i]).pow(2).sum(dim=1) / self.Rprime**2).sqrt()

            f_new = self.activate(f_tmp - grad * self.step)
            f_old, f_tmp = f_new, f_new + layer / (layer + 3) * (f_new - f_old)

        yhat = y * (f_new[:, 0])[:, None]
        for i, polygon in enumerate(self.graph.structure):
            yhat += (f_new[:, i+1])[:, None] * points[i]

        return yhat, f_new[:, 1:], f_new[:, 0]

    def decode(self, x):
        return x @ self.W

    def activate(self, x):
        m, n = x.shape
        cnt_m = torch.arange(m, device=x.device)
        cnt_n = torch.arange(n, device=x.device)
        u = x.sort(axis=1, descending=True).values
        v = (u.cumsum(dim=1) - 1) / (cnt_n + 1)
        w = v[cnt_m, (u > v).sum(dim=1) - 1]
        return (x - w.view(m, 1)).relu()

    def Phi(self, A, y, x, x0, periodic=False):

        beta = 1.0

        difference = (y.unsqueeze(1) - A.unsqueeze(0))
        if periodic:
            difference[:, :, 0] = (difference[:, :, 0] + np.pi) % (2 * np.pi) - np.pi
        weight = (difference.pow(2).sum(dim=2) / self.Rprime**2).sqrt()

        a = y[:, 0] * 0  # Zeros of the right size
        b = x0*(1)
        for i, polygon in enumerate(self.graph.structure):
            points = self.point_of_closest_approach(y, polygon)
            a += x[:, i] * ((y - points).pow(2).sum(dim=1) / self.R**2).sqrt()

        if np.isfinite(self.Rprime):

            b = (weight * x).sum(dim=1)

        return a+b

    def loss(self, A, y, x, x0, w=None, periodic=False):

        phi = self.Phi(A, y, x, x0, periodic)
        l = 0
        if w is not None:
            l = (phi * w).sum() / w.sum()
        else:
            l = phi.mean()

        return l


class LocalDictionaryLoss(torch.nn.Module):
    def __init__(self, loss):
        super(LocalDictionaryLoss, self).__init__()
        self.loss = loss

    def forward(self, A, y, x, x0, w=None, R=None):
        return self.loss(A, y, x, x0, R)
