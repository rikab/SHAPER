# SHAPER (v1.0.0)

`SHAPER` is a framework for defining, building, and evaluating generalized shape observables for collider physics, as defined in arxiv:2302:XXXX. This package can be used for evaluating an extremely large class of IRC-safe observables, with modules in place to define custom observables and jet algorithms using an intuitive geometric language. 

![3-point-ellipsiness-plus-pileup_event_0](https://user-images.githubusercontent.com/78619093/221254441-36b3bcc4-65fc-4211-aaef-2332c5dd893e.gif)

Pictured: Example of a custom jet algorithm, "3-(Ellipse+Point)iness+Pileup", as evaluated on an example top jet, as the `SHAPER` algorithm computes the value of the observable and optimal parameters.


 The `SHAPER` framework contains::
 * Pre-built observables, including [N-subjettiness](https://inspirehep.net/literature/876746) and [isotropy](https://inspirehep.net/literature/1791220).
 * Novel pre-built observables and jet algorithms for finding ring, disk, or ellipse-like jets, with optional centers for collinear radiation and optional pileup radiation
 * Modules for defining arbitrary shape observables using parameterized manifolds, and building new complex observables from old ones.
 * Modules for evaluating defined shape observables on event data, using the Sinkhorn divergence approximation of the Wasserstein metric. This returns both the shape value ("shapiness") and the optimal shape parameters.
 * Modules for visualizing shape observables and their optimization, as in the GIF above.

## Example Usage

For an example of how to use `SHAPER`, see the notebook `example.ipynb`. This notebook contains example code for loading data, using pre-built shape observables, defining custom shape observables, running the `SHAPER` algorithm to evaluate these observables, and making plots.

## Dependencies

To use `SHAPER`, the following packages must be installed as prerequisites:
* [PyTorch](https://github.com/pytorch/pytorch): A standard tensor operation library.
* [GeomLoss](https://www.kernel-operations.io/geomloss/): A library for optimal transport.
* [pyjet](https://github.com/scikit-hep/pyjet): A package for jet clustering, needed for default observable definitions.
* [Energyflow](https://energyflow.network/): A suite of particle physics tools. This package is OPTIONAL; however, many of the example datasets within `SHAPER` require this package to load. Not necessary if you provide and format your own data.
* [imageio](https://pypi.org/project/imageio/): An image manipulation package. Needed for automatic gif creation -- not needed otherwise.
* Standard python packages: [numpy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/)


## Changelog

* v1.0.0: 24 February 2023. Official public release.

Based on the work in SHAPER: Can You Hear the Shape of a Jet? (arXiv:2302:XXXX)

Bugs, Fixes, Ideas, or Questions? Contact me at rikab@mit.edu

To discuss finer mathematical details (model convergence, optimization guarantees, etc), you may also contact Akshunna S. Dogra at adogra@nyu.edu
