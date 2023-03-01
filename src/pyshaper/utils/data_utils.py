import numpy as np

# Energy-flow package for CMS Open Data loader
import energyflow as ef
from energyflow.archs import PFN, EFN
from energyflow.utils import remap_pids


# ###############################
# ########## LOAD DATA ##########
# ###############################

def load_cmsopendata(cache_dir, dataset, pt_lower, pt_upper, eta, quality, return_kfactors=True, momentum_scale=250, n=1000, amount=1, frac=1.0):

    # Load data
    specs = [f'{pt_lower} <= corr_jet_pts <= {pt_upper}', f'abs_jet_eta < {eta}', f'quality >= {quality}']
    sim = ef.mod.load(*specs, cache_dir=cache_dir, dataset=dataset, amount=amount)

    # Gen_pt for Y
    Y1 = sim.jets_f[:, sim.jet_pt]
    Y = np.zeros((Y1.shape[0], 1), dtype=np.float32)
    Y[:, 0] = Y1 / momentum_scale

    # Sim_pt for X
    X = np.zeros((Y1.shape[0], 3), dtype=np.float32)
    X[:, 0] = sim.jets_f[:, sim.jet_pt] / momentum_scale
    X[:, 1] = sim.jets_f[:, sim.jet_eta]
    X[:, 2] = sim.jets_f[:, sim.jet_phi]

    # CMS JEC's
    C = sim.jets_f[:, sim.jec]

    # PFC's
    pfcs = sim.particles

    # Shuffle and trim
    shuffle_indices = np.random.choice(np.arange(pfcs.shape[0]), size=int(pfcs.shape[0] * frac), replace=False)
    pfcs = pfcs[shuffle_indices]
    Y = Y[shuffle_indices]
    X = X[shuffle_indices]
    C = C[shuffle_indices]
    weights = sim.weights[shuffle_indices] * sim.weights.shape[0] / n

    pfcs = pfcs[:n]
    Y = Y[:n]
    X = X[:n]
    C = C[:n]
    weights = weights[:n]

    # PFC's
    events = []
    particle_counts = []

    for (i, jet) in enumerate(pfcs):

        indices = (-jet[:, 0]).argsort()
        zs = jet[indices, 0] / np.sum(jet[indices, 0])
        points = jet[indices, 1:3]

        # Center Jets
        mask = zs > 0
        yphi_avg = np.average(points[mask], weights=zs[mask], axis=0)
        points[mask] -= yphi_avg

        events.append((points, zs))
        particle_counts.append(jet.shape[0])

    particle_counts = np.array(particle_counts)

    print("Max # of particles: %d" % max(particle_counts))

    # kfactors
    if return_kfactors and dataset == "sim":
        print("test")
        return events, weights, ef.mod.kfactors('sim', sim.corr_jet_pts, sim.npvs)[shuffle_indices[:n]]
    elif return_kfactors and dataset == "gen":
        return events, weights, ef.mod.kfactors('gen', sim.jet_pts,)[shuffle_indices[:n]]

    return events, weights


def load_dataset(filename, N, mass_lower=0, mass_upper=np.inf, eta_cut=1.3, pt_cut=500, normalize=True):

    n = 0
    i = 0
    X = np.load(filename, "r",)[:N*100]
    X = np.copy(X)
    print(X.shape)

    # Normalize & Center
    events = []
    masses = []
    while n < N:
        # for x in X:
        x = X[i]
        pt = np.sum(x[:, 0])

        if normalize:
            zs = x[:, 0] / np.sum(x[:, 0])
        else:
            zs = np.copy(x[:, 0])

        mask = zs[:] > 0
        x[mask, 2] = np.unwrap(x[mask, 2])
        yphi_avg = np.average(x[mask, 1:3], weights=x[mask, 0], axis=0)
        x[mask, 1:3] -= yphi_avg

        px = np.sum(x[:, 0] * np.cos(x[:, 2]))
        py = np.sum(x[:, 0] * np.sin(x[:, 2]))
        pz = np.sum(x[:, 0]*np.sinh(x[:, 1]))
        e = np.sum(x[:, 0] * np.cosh(x[:, 1]))

        mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
        masses.append(mass)

        if mass_lower < mass < mass_upper and np.abs(yphi_avg[0]) < eta_cut and pt > pt_cut:
            events.append((x[mask, 1:3], zs[mask]))
            n += 1
        i += 1
    return events


# TODO: Parallelize
def calculate_masses(X):

    masses = []
    for x in X:
        yphi, z = x[0], x[1]
        px = np.sum(z * np.cos(yphi[:, 1]))
        py = np.sum(z * np.sin(yphi[:, 1]))
        pz = np.sum(z*np.sinh(yphi[:, 0]))
        e = np.sum(z * np.cosh(yphi[:, 0]))

        mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
        masses.append(mass)

    return np.array(masses)


def normalize_events(X, return_norm=False):

    norms = []
    Y = []
    for x in X:
        yphi, z = x[0], x[1]
        norms.append(z.sum())
        Y.append((yphi, z / z.sum()))

    if return_norm:
        return Y, norms
    else:
        return Y


def add_pileup(X, pois_mean, E_low, E_high, E_noise, R):

    Y = []
    for x in X:

        yphi, z = x[0], x[1]

        # Generate particles:
        n = np.random.poisson(pois_mean)
        e_pu = np.random.uniform(E_low, E_high)

        points = np.random.uniform(-R, R, size=(n, 2))
        weights = np.maximum(0, np.ones((n,)) * (e_pu + np.random.normal(loc=0, scale=E_noise, size=(n,))) / n)

        yphi_new = np.concatenate((yphi, points), axis=0)
        z_new = np.concatenate((z, weights), axis=0)

        Y.append((yphi_new, z_new))
    return Y
