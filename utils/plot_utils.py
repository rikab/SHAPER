import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from time import time
import os


import imageio


def plot_event(y_i, z_i, R, filename = None, color = "red", show = True):

    # plot the two events
    fig, ax = plt.subplots(constrained_layout=True, figsize = (6,6))

    pts, ys, phis = z_i, y_i[:,0], y_i[:,1]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color= color, lw=0, zorder=10, label="Event")

    # Legend 
    legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    legend.legendHandles[0]._sizes = [50]




    # plot settings
    plt.xlim(-R*1.5, R*1.5); plt.ylim(-R*1.5, R*1.5)
    plt.xlabel('Rapidity'); plt.ylabel('Azimuthal Angle')
    plt.xticks(np.linspace(-R, R, 5)); plt.yticks(np.linspace(-R, R, 5))


    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.close()
        return ax
    elif show:
        plt.show()
        return ax
    else:
        return ax
    
def plot_observable(y_i, z_i, obs, emd, title, R, filename = None, color = "red"):

    fig, ax = plt.subplots(constrained_layout=True, figsize = (6,6))

    pts, ys, phis = z_i, y_i[:,0], y_i[:,1]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color= color, lw=0, zorder=10, label="Event")
    obs.draw(ax)


    # Plot Text
    plt.text(0.05, 0.03, 'EMD: %.3f' % emd, fontsize = 10, transform = plt.gca().transAxes)


    # Legend 
    legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    legend.legendHandles[0]._sizes = [50]
    try:
        legend.legendHandles[1]._sizes = [50]
    except:
        pass


    # plot settings
    plt.title(title, loc = "right")
    plt.xlim(-R*1.25, R*1.25); plt.ylim(-R*1.25, R*1.25)
    plt.xlabel('Rapidity'); plt.ylabel('Azimuthal Angle')
    plt.xticks(np.linspace(-R, R, 5)); plt.yticks(np.linspace(-R, R, 5))


    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def make_gif(filenames, outpath, remove = True):

    # Build gif
    with imageio.get_writer(outpath, mode = "I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files:
    if remove:
        for filename in set(filenames):
            os.remove(filename)
