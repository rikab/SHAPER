
# Standard Imports
from utils.plot_utils import make_gif, plot_event, plot_observable
from geomloss import SamplesLoss
import numpy as np
from time import time
import copy
import os

# ML
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import grad

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


class Shaper(nn.Module):

    def __init__(self, observables, device=None, batch_size=None) -> None:
        super().__init__()

        # TODO: Make list, dict, or singleton
        self.observables = nn.ModuleDict(observables)
        self.observable_batch = None
        self.device = device

        # Hyperparameters
        self.batch_size = batch_size

    def forward(self, events):
        return self.calculate(events)

    def calculate(self, events, epochs=500, early_stopping=25, lr=0.05, N=250, epsilon=0.01, scaling=0.5, return_grads=False, verbose=False, plot_dictionary=None):

        self.dev = self.device
        self.reset()

        # Batch, Pad, and put into PyTorch format.
        if len(events) > 1:
            if isinstance(events[1], np.ndarray):
                events = [events, ]
        xi, ai = self.batcher(events)
        xi = xi.to(self.dev)
        ai = ai.to(self.dev)
        batch_size = xi.shape[0]

        # Initialize N_event copies of the observables (TODO: Parallelize this)
        self.observable_batch = nn.ModuleList([copy.deepcopy(self.observables) for i in range(batch_size)])
        for i in range(batch_size):
            for obs in self.observables:
                self.observable_batch[i][obs].initialize((xi[i], ai[i]))

        # Training Loop
        losses = {}
        min_losses = {}
        min_params = {}

        for obs in self.observables:

            beta = self.observables[obs].beta
            Loss = SamplesLoss("sinkhorn", p=beta, blur=epsilon**(1/beta), scaling=scaling, diameter=self.observables[obs].R * 2)

            min_losses[obs] = np.ones((batch_size,)) * np.inf
            min_params[obs] = [None, ] * batch_size
            losses[obs] = np.ones((batch_size,)) * np.inf
            gif_filenames = []

            # If gif plotter is on, save gif plots
            if plot_dictionary and plot_dictionary.get("gif_directory") is not None:
                fs = self.plot(events, obs, losses, plot_dictionary.get("gif_directory"), gif_extention="init")
                for f in fs:
                    gif_filenames.append([f])

            # If trivial parameters, skip minimization
            if len(list(self.observables[obs].parameters())) == 0 or epochs == 0:

                # Sample observables (TODO: parallelize sampling over batch)

                samples = [self.observable_batch[i][obs].sample(N) for i in range(batch_size)]
                yj = [sample[0] for sample in samples]
                bj = [sample[1] for sample in samples]
                yj, bj = nn.utils.rnn.pad_sequence(yj, batch_first=True), nn.utils.rnn.pad_sequence(bj, batch_first=True)

                yj.to(self.dev)
                bj.to(self.dev)

                min_losses[obs] = Loss(ai, xi, bj, yj).cpu().detach().numpy() / self.observables[obs].R**beta
                for i in np.array(range(batch_size)):
                    min_params[obs][i] = self.observable_batch[i][obs].get_dict(self.device)

                if verbose:
                    print("Observable:", obs, "Mean Loss =", np.mean(min_losses[obs]))

            else:

                # Optimizer
                optimizer = optim.Adam(self.observable_batch.parameters(), lr=lr)

                if verbose:
                    t1 = time()

                # Early Stopping Counters
                counts = np.zeros((batch_size,), dtype=np.int32)
                mask = counts < early_stopping

                for epoch in range(epochs):

                    # Sample observables (TODO: parallelize sampling over batch)
                    samples = [self.observable_batch[i][obs].sample(N) for i in np.array(range(batch_size))[mask]]
                    yj = [sample[0] for sample in samples]
                    bj = [sample[1] for sample in samples]
                    yj, bj = nn.utils.rnn.pad_sequence(yj, batch_first=True), nn.utils.rnn.pad_sequence(bj, batch_first=True)

                    yj.to(self.dev)
                    bj.to(self.dev)

                    # Calculate losses
                    optimizer.zero_grad()
                    # for param in self.observable_batch.parameters():
                    #     param.grad = None
                    #     param.requires_grad = True
                    Loss_xy = Loss(ai[mask], xi[mask], bj, yj) / self.observables[obs].R**beta
                    Loss_xy.sum().backward()
                    optimizer.step()

                    losses[obs][mask] = Loss_xy.cpu().detach().numpy()

                    # Enforce parameter constraints (TODO: Parallelize)
                    for i in range(batch_size):
                        self.observable_batch[i][obs].enforce()

                    # If gif plotter is on, save gif plots
                    if plot_dictionary and plot_dictionary.get("gif_directory") is not None:
                        fs = self.plot(events, obs, losses, plot_dictionary.get("gif_directory"), gif_extention="%d" % epoch)
                        for i in range(batch_size):
                            gif_filenames[i].append(fs[i])

                    if verbose:
                        print("Observable:", obs, "Epoch %d" % epoch, "Mean Loss =", np.mean(losses[obs]), "Elapsed time = %.3fs" % (
                            time() - t1), "Percentage done = %.3f " % (100 * (batch_size - np.sum(mask))/batch_size))

                    # Early Stopping Calculation (TODO: Parallelize)
                    for i in np.array(range(batch_size))[mask]:
                        if losses[obs][i] < min_losses[obs][i] * (1.0-epsilon):
                            min_losses[obs][i] = losses[obs][i]
                            d = self.observable_batch[i][obs].get_dict(self.device)
                            d["EMD"] = min_losses[obs][i]
                            min_params[obs][i] = d
                            counts[i] = 0
                        else:
                            counts[i] += 1
                    mask = counts < early_stopping
                    if np.sum(mask) == 0:
                        break

                    if (100 * (batch_size - np.sum(mask))/batch_size) > 95:
                        break

            # Plotting
            if plot_dictionary and plot_dictionary.get("plot_directory") is not None:
                self.plot(events, obs, losses, plot_dictionary.get("plot_directory"), extension=plot_dictionary["extension"], title=plot_dictionary["title"])
            if plot_dictionary and plot_dictionary.get("gif_directory") is not None:
                self.make_gifs(gif_filenames, obs, plot_dictionary.get("gif_directory"))

        # format params
        params = {}
        for obs in self.observables:
            pass

        # Calculate gradients
        if return_grads:

            Fs = {}
            dxs = {}

            for obs in self.observables:

                ai.requires_grad = True
                xi.requires_grad = True

                beta = self.observables[obs].beta
                Loss = SamplesLoss("sinkhorn", p=beta, blur=epsilon**(1/beta), scaling=scaling, diameter=self.observables[obs].R * 2)

                samples = [self.observable_batch[i][obs].sample(N) for i in range(batch_size)]
                yj = [sample[0] for sample in samples]
                bj = [sample[1] for sample in samples]
                yj, bj = nn.utils.rnn.pad_sequence(yj, batch_first=True), nn.utils.rnn.pad_sequence(bj, batch_first=True)

                yj.to(self.dev)
                bj.to(self.dev)

                Loss_xy = Loss(ai, xi, bj, yj).sum()
                F_i, dx_i = grad(Loss_xy, [ai, xi])

                Fs[obs] = F_i.cpu().detach().numpy()
                dxs[obs] = dx_i.cpu().detach().numpy()

            return min_losses, min_params, Fs, dxs

        else:
            return min_losses, min_params

    def plot(self, events, obs, losses, directory, title="", extension="png", gif_extention=""):

        # Make directory is it doesn't exist
        dir = os.path.join(directory, obs)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("Created Directory: %s" % dir)

        filenames = []
        for i in range(len(events)):

            filename = os.path.join(dir, "%s event_%d.%s" % (obs, i, extension))
            _title = "%s, %s" % (obs, title)
            if gif_extention != "":
                filename = os.path.join(dir, "event_%d_%s.gif.png" % (i, gif_extention))
                _title += ", Epoch %s" % (str(gif_extention))
            plot_observable(events[i][0], events[i][1], self.observable_batch[i][obs], losses[obs][i], title=_title, R=self.observables[obs].R, filename=filename)
            filenames.append(filename)
        return filenames

    def make_gifs(self, filenames, obs, directory):

        # Make directory is it doesn't exist
        dir = os.path.join(directory, obs)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("Created Directory: %s" % dir)

        for i in range(len(filenames)):

            filename = os.path.join(dir, "%s event_%d.gif" % (obs, i))
            make_gif(filenames[i], filename)

    # Function to format events of the form [(x, a)] into a padded pytorch tensor of shape (Batch_size, Pad, dim)

    def batcher(self, events, batch_size=None):

        if batch_size is None:

            zs = [torch.Tensor(event[1]) for event in events]
            points = [torch.Tensor(event[0]) for event in events]
            return nn.utils.rnn.pad_sequence(points, batch_first=True), nn.utils.rnn.pad_sequence(zs, batch_first=True)

        else:
            raise NotImplementedError("Custom batch sizes not yet implemented")

    def reset(self):
        self.observable_batch = None
        for obs in self.observables:
            if len(list(self.observables[obs].parameters())) > 0:
                self.dev = next(self.parameters()).device

    def format_param_dictionary(self, dictionary):

        new_dict = {}
        for obs in self.observables:
            pass
