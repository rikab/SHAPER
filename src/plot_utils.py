# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
from src.Graph import Background
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
import imageio
from time import time


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import energyflow as ef
from energyflow.archs import PFN, EFN
from energyflow.utils import remap_pids



def plot(y_i, z_i, y_j, z_j, x_ij, loss,  R, z0, filename, name, shapes, color = "red"):


    # plot the two events
    fig, ax = plt.subplots(constrained_layout=True, figsize = (8,8))

    # plot data points
    if color is None:
        color = "red"
    pts, ys, phis = z_i, y_i[:,0], y_i[:,1]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500, color= color, lw=0, zorder=10, label="Event")

    flag = True
    for shape in shapes:

        pts, ys, phis = shape.get_zs(), shape.get_points()[:,0], shape.get_points()[:,1]
        if type(shape) is Background:
            ax.scatter(ys, phis, marker='o', s=2 * pts * 500 , color="royalblue", lw=0, zorder=10, label="Background")
        elif flag:
            ax.scatter(ys, phis, marker='o', s=2 * pts * 500 , color="purple", lw=0, zorder=10, label="Structure Points")
            flag = False

    # colors = ['red', 'purple']
    # labels = ['Event', 'Structure Points']
    # z = [z_i, z_j]
    # for i,ev in enumerate([y_i, y_j]):
    #     pts, ys, phis = z[i], ev[:,0], ev[:,1]
    #     ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts) , color=colors[i], lw=0, zorder=10, label=labels[i])


    # plot the flow
    mx = 10 * x_ij.max()
    xs, xt = y_i[:,0:2], y_j[:,0:2]
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if x_ij[i, j] > 0:
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        alpha=x_ij[i, j]/mx, lw=1.25, color='black')

    # plot settings
    plt.xlim(-R*1.5, R*1.5); plt.ylim(-R*1.5, R*1.5)
    plt.xlabel('Rapidity'); plt.ylabel('Azimuthal Angle')
    plt.title(name)
    plt.xticks(np.linspace(-R, R, 5)); plt.yticks(np.linspace(-R, R, 5))

    emd_val =  loss 
    plt.text(0.05, 0.03, 'EMD: {:.3f} GeV'.format(emd_val), fontsize=10, transform=plt.gca().transAxes)
    for i, shape in enumerate(shapes):
        plt.text(0.05, 0.06 + 0.03*i, str(shape), fontsize=10, transform=plt.gca().transAxes)
    plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)

    for shape in shapes:
        shape.draw(ax)


    ax.set_aspect('equal')
    
    plt.savefig(filename)
    plt.close()


def plot_event(y_i, z_i, R, filename = None, graph = None, A = None, color = "red"):

    # plot the two events
    fig, ax = plt.subplots(constrained_layout=True)

    colors = ['red', 'purple']
    labels = ['Event', 'Structure Points']
    z = [z_i]
    for i,ev in enumerate([y_i]):
        pts, ys, phis = z[i], ev[:,0], ev[:,1]
        ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts) , color=color, lw=0, zorder=10, label=labels[i])


    # # plot the flow
    # mx = 10 * x_ij.max()
    # xs, xt = y_i[:,0:2], y_j[:,0:2]
    # for i in range(xs.shape[0]):
    #     for j in range(xt.shape[0]):
    #         if x_ij[i, j] > 0:
    #             ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
    #                     alpha=x_ij[i, j]/mx, lw=1.25, color='black')

    # plot settings
    plt.xlim(-R*1.5, R*1.5); plt.ylim(-R*1.5, R*1.5)
    plt.xlabel('Rapidity'); plt.ylabel('Azimuthal Angle')
    plt.xticks(np.linspace(-R, R, 5)); plt.yticks(np.linspace(-R, R, 5))

    # emd_val = np.sqrt(np.sum(z_i) * np.sum(z_j)) * loss + np.abs(np.sum(z_i) - np.sum(z_j))
    # plt.text(0.6, 0.03, 'EMD: {:.3f} GeV'.format(emd_val), fontsize=10, transform=plt.gca().transAxes)
    # plt.legend(loc=(0.1, 1.0), frameon=False, ncol=2, handletextpad=0)

    # if A is not None:
    #     graph.draw_polygons(ax, A)

    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def make_gif(filenames, outpath, remove = True):

    # build gif
    with imageio.get_writer(outpath, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    if remove:
        for filename in set(filenames):
            os.remove(filename)


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def load_data(pt_lower, pt_upper, eta, quality, pad, x_dim = 3, momentum_scale = 250, n = 100000, amount = 1, max_particle_select = None, frac = 1.0):

    # Load data
    specs = [f'{pt_lower} <= gen_jet_pts <= {pt_upper}', f'abs_jet_eta < {eta}', f'quality >= {quality}']
    sim = ef.mod.load(*specs, dataset='sim', amount= amount)

    # Gen_pt for Y
    Y1 = sim.jets_f[:,sim.gen_jet_pt]
    Y = np.zeros((Y1.shape[0], 1))
    Y[:,0] = Y1 / momentum_scale

    # Sim_pt for X
    X = np.zeros((Y1.shape[0],3))
    X[:,0] = sim.jets_f[:,sim.jet_pt] / momentum_scale
    X[:,1] = sim.jets_f[:,sim.jet_eta]
    X[:,2] = sim.jets_f[:,sim.jet_phi]

    # CMS JEC's
    C = sim.jets_f[:,sim.jec]

    # PFC's
    pfcs = sim.particles
    dataset = np.zeros( (pfcs.shape[0], pad, x_dim) )
    particle_counts = []
    print(pfcs.size)
    for (i, jet) in enumerate(pfcs):
        size = min(jet.shape[0], pad)
        indices = (-jet[:,0]).argsort()
        dataset[i, :size, 0] = jet[indices[:size],0] / momentum_scale
        dataset[i, :size, 1] = jet[indices[:size],1]
        dataset[i, :size, 2] = jet[indices[:size],2]
        if x_dim == 4:
            dataset[i, :size, 3] = jet[indices[:size],4] # PID
        particle_counts.append(jet.shape[0])
    if x_dim == 4:
        remap_pids(dataset, pid_i = 3, error_on_unknown = False)
    particle_counts = np.array(particle_counts)

    for x in dataset:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights = x[mask,0], axis = 0)
        x[mask,1:3] -= yphi_avg  


    # Trim and shuffle

    if max_particle_select is not None:
        dataset = dataset[particle_counts < max_particle_select]
        Y = Y[particle_counts < max_particle_select]
        X = X[particle_counts < max_particle_select]
        C = C[particle_counts < max_particle_select]
        particle_counts = particle_counts[particle_counts < max_particle_select]

    shuffle_indices = np.random.choice(np.arange(dataset.shape[0]), size = int(dataset.shape[0] * frac), replace=False)

    dataset = dataset[shuffle_indices, :, :]
    Y = Y[shuffle_indices]
    X = X[shuffle_indices]
    C = C[shuffle_indices]
    particle_counts = particle_counts[shuffle_indices]

    dataset = dataset[:n]
    Y = Y[:n]
    X = X[:n]
    C = C[:n]
    particle_counts = particle_counts[:n]


    print(X.shape, dataset.shape)
    print("Max # of particles: %d" % max(particle_counts))
    return X, dataset, Y, C, particle_counts