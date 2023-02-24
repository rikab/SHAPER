# SHAPER
## v1.0.0

![3-point-ellipsiness-plus-pileup_event_0](https://user-images.githubusercontent.com/78619093/221254441-36b3bcc4-65fc-4211-aaef-2332c5dd893e.gif)

Pictured: Example of a custom jet algorithm, "3-(Ellipse+Point)iness+Pileup", as evaluated on an example top jet, as the `SHAPER` algorithm computes the value of the observable and optimal parameters.



Python implementation of the `SHAPER` algorithm for computing arbitrary shape observables for collider physics, as defined in arxiv:2302:XXXX. The `SHAPER` algorithm contains modules for defining shape observables using parameterized manifolds, and for using the Sinkhorn divergence to evaluate these observables on data.


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

Bugs, Fixes, or Questions? Contact me at rikab@mit.edu
