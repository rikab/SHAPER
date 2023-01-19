# Standard Imports
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle as pltRectangle
from matplotlib.patches import Ellipse as pltEllipse
from matplotlib.patches import Circle as pltCircle
from src.Manifolds import Coordinates2D, Simplex, PositiveReals, Circle
import torch.nn as nn
import torch
from src.Observables import Observable
import numpy as np
from time import time
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')


def buildCommmonObservables(N, beta, R, device):

    # Sample at N weighted Dirac deltas
    def point_sampler(N, param_dict):
        return (param_dict["Points"].params, param_dict["Weights"].params)

    def point_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple", label="Disk", marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.75, zorder=15, lw=3)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f" % (centers[i, 0], centers[i, 1], weights[i]), fontsize=18, transform=plt.gca().transAxes)

    # Disk

    def ring_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = points.T
        z = torch.cat([weights[i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        return (e, z)

    def ring_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], fill=False, edgecolor="purple", alpha=0.5)
            ax.add_artist(draw_circle)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f, Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Disk

    def disk_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = points.T
        z = torch.cat([weights[i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        return (e, z)

    def disk_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f, Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Disk

    def ellipse_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii1 = param_dict["Radius1"].params
        radii2 = param_dict["Radius2"].params
        angles = param_dict["Angles"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([radii1[:, None] * torch.cos(phi + angles[:, None]), radii2[:, None] * torch.sin(phi + angles[:, None])], axis=1) * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = points.T
        z = torch.cat([weights[i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        return (e, z)

    def ellipse_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii1 = param_dict["Radius1"].params.clone().detach().numpy()
        radii2 = param_dict["Radius2"].params.clone().detach().numpy()
        angles = param_dict["Angles"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltEllipse(centers[i, :], 2*radii1[i], 2*radii2[i], angle=angles[i] * 180 / np.pi, facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Text
            eccentricity = np.sqrt(1 - min(radii1[i], radii2[i]) / max(radii1[i], radii2[i]))
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f, Rad: %.2f, Ecc. %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], np.sqrt(radii1[i] * radii2[i]), eccentricity), fontsize=18, transform=plt.gca().transAxes)

    # Point plus disk

    def point_ring_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = torch.cat([centers, points.T], dim=0)
        z1 = torch.cat([weights[i] * torch.ones((1,), device=device) for i in range(num)], dim=0)
        z2 = torch.cat([weights[num + i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        z = torch.cat([z1, z2], dim=0)
        return (e, z)

    def point_ring_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], fill=False, edgecolor="purple", alpha=0.5)
            ax.add_artist(draw_circle)

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple",  marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.5, zorder=15, lw=3)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z$_\delta$, z$_\mathcal{O}$: (%.2f, %.2f), Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], weights[i+num], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Point plus disk

    def point_disk_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = torch.cat([centers, points.T], dim=0)
        z1 = torch.cat([weights[i] * torch.ones((1,), device=device) for i in range(num)], dim=0)
        z2 = torch.cat([weights[num + i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        z = torch.cat([z1, z2], dim=0)
        return (e, z)

    def point_disk_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple",  marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.5, zorder=15, lw=3)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z$_\delta$, z$_\mathcal{O}$: (%.2f, %.2f), Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], weights[i+num], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Point plus disk

    def point_ellipse_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii1 = param_dict["Radius1"].params
        radii2 = param_dict["Radius2"].params
        angles = param_dict["Angles"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([radii1[:, None] * torch.cos(phi + angles[:, None]), radii2[:, None] * torch.sin(phi + angles[:, None])], axis=1) * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = torch.cat([centers, points.T], dim=0)
        z1 = torch.cat([weights[i] * torch.ones((1,), device=device) for i in range(num)], dim=0)
        z2 = torch.cat([weights[num + i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        z = torch.cat([z1, z2], dim=0)
        return (e, z)

    def point_ellipse_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii1 = param_dict["Radius1"].params.clone().detach().numpy()
        radii2 = param_dict["Radius2"].params.clone().detach().numpy()
        angles = param_dict["Angles"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltEllipse(centers[i, :], 2*radii1[i], 2*radii2[i], angle=angles[i] * 180 / np.pi, facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple",  marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.5, zorder=15, lw=3)

            # Text
            eccentricity = np.sqrt(1 - min(radii1[i], radii2[i]) / max(radii1[i], radii2[i]))
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.15 + 0.10*i, s + r"x: (%.2f, %.2f), z$_\delta$, z$_\mathcal{O}$: (%.2f, %.2f)" %
                     (centers[i, 0], centers[i, 1], weights[i], weights[i+num], ), fontsize=18, transform=plt.gca().transAxes)
            plt.text(0.05, 0.10 + 0.10*i, r"    Eff. Rad: %.2f, Ecc: %.2f" % (np.sqrt(radii1[i] * radii2[i]), eccentricity), fontsize=18, transform=plt.gca().transAxes)

    # Sample from a normalized uniform distribution

    def uniform_sampler(N, param_dict):
        points = torch.FloatTensor(N, 2).uniform_(-R, R).to(device)

        zs = torch.ones((N,)).to(device) / N
        return (points, zs)

    # Sample at N weighted Dirac deltas

    def point_sampler(N, param_dict):
        return (param_dict["Points"].params, param_dict["Weights"].params)

    def point_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple", label="Disk", marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.75, zorder=15, lw=3)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f" % (centers[i, 0], centers[i, 1], weights[i]), fontsize=18, transform=plt.gca().transAxes)

    # Disk

    def ring_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = points.T
        z = torch.cat([weights[i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        return (e, z)

    def ring_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], fill=False, edgecolor="purple", alpha=0.5)
            ax.add_artist(draw_circle)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f, Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Disk

    def disk_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = points.T
        z = torch.cat([weights[i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        return (e, z)

    def disk_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f, Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Disk

    def ellipse_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii1 = param_dict["Radius1"].params
        radii2 = param_dict["Radius2"].params
        angles = param_dict["Angles"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([radii1[:, None] * torch.cos(phi + angles[:, None]), radii2[:, None] * torch.sin(phi + angles[:, None])], axis=1) * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = points.T
        z = torch.cat([weights[i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        return (e, z)

    def ellipse_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii1 = param_dict["Radius1"].params.clone().detach().numpy()
        radii2 = param_dict["Radius2"].params.clone().detach().numpy()
        angles = param_dict["Angles"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltEllipse(centers[i, :], 2*radii1[i], 2*radii2[i], angle=angles[i] * 180 / np.pi, facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Text
            eccentricity = np.sqrt(1 - min(radii1[i], radii2[i]) / max(radii1[i], radii2[i]))
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z: %.2f, Rad: %.2f, Ecc. %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], np.sqrt(radii1[i] * radii2[i]), eccentricity), fontsize=18, transform=plt.gca().transAxes)

    # Point plus disk

    def point_ring_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = torch.cat([centers, points.T], dim=0)
        z1 = torch.cat([weights[i] * torch.ones((1,), device=device) for i in range(num)], dim=0)
        z2 = torch.cat([weights[num + i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        z = torch.cat([z1, z2], dim=0)
        return (e, z)

    def point_ring_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], fill=False, edgecolor="purple", alpha=0.5)
            ax.add_artist(draw_circle)

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple",  marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.5, zorder=15, lw=3)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z$_\delta$, z$_\mathcal{O}$: (%.2f, %.2f), Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], weights[i+num], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Point plus disk

    def point_disk_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii = param_dict["Radius"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([torch.cos(phi), torch.sin(phi)], axis=1) * radii[:, None, None] * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = torch.cat([centers, points.T], dim=0)
        z1 = torch.cat([weights[i] * torch.ones((1,), device=device) for i in range(num)], dim=0)
        z2 = torch.cat([weights[num + i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        z = torch.cat([z1, z2], dim=0)
        return (e, z)

    def point_disk_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii = param_dict["Radius"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltCircle(centers[i, :], radii[i], facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple",  marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.5, zorder=15, lw=3)

            # Text
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.10 + 0.05*i, s + r"x: (%.2f, %.2f), z$_\delta$, z$_\mathcal{O}$: (%.2f, %.2f), Rad: %.2f" %
                     (centers[i, 0], centers[i, 1], weights[i], weights[i+num], radii[i]), fontsize=18, transform=plt.gca().transAxes)

    # Point plus disk

    def point_ellipse_sampler(N, param_dict):

        centers = param_dict["Points"].params
        num = param_dict["Points"].N
        radii1 = param_dict["Radius1"].params
        radii2 = param_dict["Radius2"].params
        angles = param_dict["Angles"].params
        weights = param_dict["Weights"].params

        phi = 2 * np.pi * torch.rand(num, N).to(device)
        r = torch.sqrt(torch.rand(num, N)).to(device)
        points = torch.stack([radii1[:, None] * torch.cos(phi + angles[:, None]), radii2[:, None] * torch.sin(phi + angles[:, None])], axis=1) * r[:, None, :] + centers[:, :, None]
        points = torch.cat([point for point in points], dim=1)

        # Concatenate and reweight
        e = torch.cat([centers, points.T], dim=0)
        z1 = torch.cat([weights[i] * torch.ones((1,), device=device) for i in range(num)], dim=0)
        z2 = torch.cat([weights[num + i] * torch.ones((N,), device=device) / N for i in range(num)], dim=0)
        z = torch.cat([z1, z2], dim=0)
        return (e, z)

    def point_ellipse_plotter(ax, param_dict):

        centers = param_dict["Points"].params.clone().detach().numpy()
        radii1 = param_dict["Radius1"].params.clone().detach().numpy()
        radii2 = param_dict["Radius2"].params.clone().detach().numpy()
        angles = param_dict["Angles"].params.clone().detach().numpy()
        weights = param_dict["Weights"].params.clone().detach().numpy()
        num = param_dict["Points"].N

        for i in range(num):
            # Circle
            draw_circle = pltEllipse(centers[i, :], 2*radii1[i], 2*radii2[i], angle=angles[i] * 180 / np.pi, facecolor="purple", edgecolor="purple", alpha=0.25)
            ax.add_artist(draw_circle)

            # Center
            ax.scatter(centers[i, 0], centers[i, 1], color="Purple",  marker="x", s=2 * weights[i] * 500/np.sum(weights), alpha=0.5, zorder=15, lw=3)

            # Text
            eccentricity = np.sqrt(1 - min(radii1[i], radii2[i]) / max(radii1[i], radii2[i]))
            if num > 1:
                s = "%d) " % (num - i)
            else:
                s = ""
            plt.text(0.05, 0.15 + 0.10*i, s + r"x: (%.2f, %.2f), z$_\delta$, z$_\mathcal{O}$: (%.2f, %.2f)" %
                     (centers[i, 0], centers[i, 1], weights[i], weights[i+num], ), fontsize=18, transform=plt.gca().transAxes)
            plt.text(0.05, 0.10 + 0.10*i, r"    Eff. Rad: %.2f, Ecc: %.2f" % (np.sqrt(radii1[i] * radii2[i]), eccentricity), fontsize=18, transform=plt.gca().transAxes)

    observables_array = []
    CommonObservables = {}

    for m in range(N):

        n = m + 1
        _nsubjettiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(n)}, point_sampler, beta=1, R=R, initializer="kt", plotter=point_plotter)

        _nringiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(n), "Radius": PositiveReals(n, 0)},
                                 ring_sampler, beta=1, R=R, initializer="kt", plotter=ring_plotter)
        _ndiskiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(n), "Radius": PositiveReals(n, 0)},
                                 disk_sampler, beta=1, R=R, initializer="kt", plotter=disk_plotter)
        _nellipsiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(n), "Radius1": PositiveReals(n, 0), "Radius2": PositiveReals(n, 0),
                                    "Angles": Circle(n, R/2)}, ellipse_sampler, beta=1, R=R, initializer="kt", plotter=ellipse_plotter)

        _npoint_ringiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(2*n), "Radius": PositiveReals(n, 0)},
                                       point_ring_sampler, beta=1, R=R, initializer="kt", plotter=point_ring_plotter)
        _npoint_diskiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(2*n), "Radius": PositiveReals(n, 0)},
                                       point_disk_sampler, beta=1, R=R, initializer="kt", plotter=point_disk_plotter)
        _npoint_ellipseiness = Observable({"Points": Coordinates2D(n), "Weights": Simplex(2*n), "Radius1": PositiveReals(n, 0), "Radius2": PositiveReals(n,
                                                                                                                                                         0), "Angles": Circle(n, 0)}, point_ellipse_sampler, beta=1, R=R, initializer="kt", plotter=point_ellipse_plotter)

        observables_array.append(_nsubjettiness)
        observables_array.append(_nringiness)
        observables_array.append(_ndiskiness)
        observables_array.append(_nellipsiness)
        observables_array.append(_npoint_ringiness)
        observables_array.append(_npoint_diskiness)
        observables_array.append(_npoint_ellipseiness)

        CommonObservables["%d-Subjettiness" % (n)] = _nsubjettiness
        CommonObservables["%d-Ringiness" % (n)] = _nringiness
        CommonObservables["%d-Diskiness" % (n)] = _ndiskiness
        CommonObservables["%d-Ellipsiness" % (n)] = _nellipsiness
        CommonObservables["%d-Point-Ringiness" % (n)] = _npoint_ringiness
        CommonObservables["%d-Point-Diskiness" % (n)] = _npoint_diskiness
        CommonObservables["%d-Point-Ellipsiness" % (n)] = _npoint_ellipseiness

    return CommonObservables, observables_array
