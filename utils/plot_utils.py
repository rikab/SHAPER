import imageio
import os
from time import time
import numpy as np
from matplotlib import pyplot as plt


def plot_event(y_i, z_i, R, filename=None, color="red", title="", show=True):

    # plot the two events
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))

    pts, ys, phis = z_i, y_i[:, 0], y_i[:, 1]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color=color, lw=0, zorder=10, label="Event")

    # Legend
    # legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    # legend.legendHandles[0]._sizes = [150]

    # plot settings
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.xlabel('Rapidity')
    plt.ylabel('Azimuthal Angle')
    plt.title(title)
    plt.xticks(np.linspace(-R, R, 5))
    plt.yticks(np.linspace(-R, R, 5))

    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.show()
        plt.close()
        return ax
    elif show:
        plt.show()
        return ax
    else:
        return ax


def plot_observable(y_i, z_i, obs, emd, title, R, filename=None, color="red"):

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))

    pts, ys, phis = z_i, y_i[:, 0], y_i[:, 1]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color=color, lw=0, zorder=3, label="Event")
    obs.draw(ax)

    # Plot Text
    plt.text(0.05, 0.05, 'EMD: %.3f' % emd, fontsize=18, transform=plt.gca().transAxes)

    # Legend
    # legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    # legend.legendHandles[0]._sizes = [50]
    # try:
    #     # legend.legendHandles[1]._sizes = [50]
    # except:
    #     pass

    # plot settings
    plt.title(title, loc="right")
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.xlabel('Rapidity')
    plt.ylabel('Azimuthal Angle')
    plt.xticks(np.linspace(-R, R, 5))
    plt.yticks(np.linspace(-R, R, 5))

    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def make_gif(filenames, outpath, remove=True):

    # Build gif
    with imageio.get_writer(outpath, mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files:
    if remove:
        for filename in set(filenames):
            os.remove(filename)


def plot_hist(axes, data_arr, data_weights, sim_arr, sim_weights, lims, color="red", label="", obs="", bins=15):

    n = len(data_arr)
    counts, bin_edges = np.histogram(data_arr, bins=bins, range=lims, weights=data_weights[:n])
    sim_counts, sim_egdges = np.histogram(sim_arr, bins=bins, range=lims, weights=sim_weights[:n])
    step = bin_edges[1] - bin_edges[0]
    errs2 = np.histogram(data_arr, bins=bins, weights=data_weights[:n]*data_weights[:n])[0]
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    axes[0].errorbar(bin_centres, counts, yerr=np.sqrt(errs2), xerr=step / 2, fmt='o', color=color, label=r"Data " + label)
    axes[0].hist(sim_arr, bins=bins, weights=sim_weights[:n], range=lims, color=color, alpha=0.25, lw=3, histtype="step", label=r"SIM " + label)

    reg = 1e-6
    axes[1].errorbar(bin_centres, counts / (sim_counts + reg), xerr=step / 2, yerr=np.sqrt(errs2) / (sim_counts + reg), color=color, fmt='o')

    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, bottom=True, left=True, direction='in', which='both')

    # Formatting and labels
    xlabel = obs
    ylabel = 'Differential Cross Section [nb]'

    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel("Data / Sim")

    # Set axes limits
    y_limit = axes[0].get_ylim()[1]
    y_limit = max(y_limit, 1.25 * np.max(counts))
    axes[0].set_ylim([0.0, y_limit])
    axes[1].set_ylim([0.5, 1.5])
    axes[1].axhline(1.0, color="black", ls="--")

    axes[0].set_title('EMD Distributions - CMS Open Data', loc="right")
    axes[0].legend()

    return counts, bin_edges
