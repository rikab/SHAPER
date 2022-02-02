import numpy as np
import scipy.io as sio

import keras
from keras.datasets import mnist

from sklearn.datasets import make_moons

import torch
from torchvision import datasets, transforms





# example usage:

# from datasets import get_mnist_data, get_yale_data, get_hyperspectral_data
# x, y = get_mnist_data()
# x, y = get_yale_data()
# x = get_hyperspectral_data()


def prune(data, labels, wanted):
    n = len(data)
    idx = [i for i in range(n) if labels[i] in wanted]
    data = data[idx]
    labels = labels[idx]
    lookup = np.arange(int(labels.max() + 1))
    lookup[wanted] = np.arange(len(wanted))
    labels = lookup[labels]
    return data, labels


def generate_background(y_i, z_i, N, percentage):

    # Add some background
    bg = np.random.random((N,2)) * 3 - 1.5
    bgz = percentage * np.ones((N)) / (N)
    y_i = np.concatenate((y_i, bg))
    z_i = np.concatenate((z_i * (1-percentage), bgz)) / np.sum(np.concatenate((z_i * (1-percentage), bgz)))
    return y_i, z_i


def generate_ring_like_event(N_sig, N_bg, percent_bg, center, radius, width):

    phi = np.random.random(N_sig) * 2 * np.pi
    r = np.abs((np.random.normal(size = N_sig))*width + radius)
    y_i = np.zeros((N_sig,2))
    y_i[:,0] = r * np.cos(phi)
    y_i[:,1] = r * np.sin(phi)
    y_i = y_i + center
    z_i = np.ones((N_sig)) / (N_sig)
    return generate_background(y_i, z_i, N_bg, percent_bg)


def generate_circle_like_event(N_sig, N_bg, percent_bg, center, radius):

    y_i = np.random.normal(size=(N_sig, 2)) * radius + center
    z_i = np.ones((N_sig)) / (N_sig)
    return generate_background(y_i, z_i, N_bg, percent_bg)

def get_data(dataset, path, quantity=100):



    if dataset == "line":
        data = np.zeros((quantity, 2))
        data[:,0] = np.random.uniform(-1, 1, size = quantity)
        data[:,1] = np.random.normal(scale = 0.1, size = quantity)
        data += 0.1 * np.random.normal(size = data.shape)
        
        
        
        labels = np.ones(data.shape[0]) / data.shape[0]
        return data, labels, 2
    
    if dataset == 'graphs':
        data = np.loadtxt(open("/home/rikab/Documents/Physics/Grad/Research/Simplex/Datasets/graph_data.csv", "rb"), delimiter=",",  skiprows=1) / 250
        data = torch.tensor(data).float()
        labels = torch.tensor(np.ones(data.shape[0]))
        return data, labels, 2

    if dataset == "moons":
        data, labels = make_moons(quantity)
        data[:, 0] += 0
        data[:, 1] += 0
        data = (data - np.mean(data)) / np.std(2 * data)
        data += 0.1 * np.random.normal(size = data.shape)
        labels = np.ones(data.shape[0]) / data.shape[0]
        return data, labels, 2
    if dataset == 'energies':
        samples = np.load("/home/rikab/Documents/Physics/Grad/Research/Simplex/Datasets/ZJets/Event_%d.npy" % 2,  allow_pickle=True)
        # samples = samples[samples[:, 1] > 0]
        data, weights = samples[..., 1:], samples[..., 0]
        labels = torch.ones_like(torch.tensor(weights), dtype=torch.int32)
        return torch.tensor(data).float(), torch.tensor(weights).float(), 2
    elif dataset == "mnist":
        data, labels = get_mnist_data()
        data, labels = prune(data, labels, [0, 3, 4, 6, 7])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 5
    elif dataset == "salinas":
        data = get_hyperspectral_data(f"{path}/SalinasA_smallNoise.mat")
        labels = get_hyperspectral_data(f"{path}/SalinasA_gt.mat")
        data, labels = prune(data, labels, [0, 1, 10, 11, 12, 13, 14])
        data = torch.tensor(data).float()
        labels = torch.tensor(labels)
        data = data.reshape(86, 83, -1).permute(1, 0, 2).reshape(83 * 86, -1)
        data -= data.min()
        data /= data.max()
        data, labels = data[np.where(labels)], labels[np.where(labels)] - 1
        return data, labels, 6
    elif dataset == "yale2":
        data, labels = get_yale_data() if path is None else get_yale_data(path)
        data, labels = prune(data, labels, [4, 8])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 2
    elif dataset == "yale3":
        data, labels = get_yale_data() if path is None else get_yale_data(path)
        data, labels = prune(data, labels, [4, 8, 20])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 3
    else:
        print(f"unknown dataset '{dataset}")


def main():

    pass

if __name__ == '__main__':
    main()
