# SHAPER (v1.1.1)

[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/rikab/shaper)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7689890.svg)](https://doi.org/10.5281/zenodo.7689890)

[![PyPI version](https://img.shields.io/pypi/v/pyshaper.svg)](https://pypi.org/project/pyshaper/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/pyshaper.svg)](https://pypi.org/project/pyshaper/)

`SHAPER` is a framework for defining, building, and evaluating generalized shape observables for collider physics, as defined in ["SHAPER: Can You Hear the Shape of a Jet?" (arxiv:2302.12266)](https://arxiv.org/abs/2302.12266). This package can be used for evaluating an extremely large class of IRC-safe observables, with modules in place to define custom observables and jet algorithms using an intuitive geometric language.

![3-point-ellipsiness-plus-pileup_event_0](https://user-images.githubusercontent.com/78619093/221254441-36b3bcc4-65fc-4211-aaef-2332c5dd893e.gif)

Pictured: Example of a custom jet algorithm, "3-(Ellipse+Point)iness+Pileup", as evaluated on an example top jet, as the `SHAPER` algorithm computes the value of the observable and optimal parameters.

The `SHAPER` framework contains::

- Pre-built observables, including [N-subjettiness](https://inspirehep.net/literature/876746) and [isotropy](https://inspirehep.net/literature/1791220).
- Novel pre-built observables and jet algorithms for finding ring, disk, or ellipse-like jets, with optional centers for collinear radiation and optional pileup radiation
- Modules for defining arbitrary shape observables using parameterized manifolds, and building new complex observables from old ones.
- Modules for evaluating defined shape observables on event data, using the Sinkhorn divergence approximation of the Wasserstein metric. This returns both the shape value ("shapiness") and the optimal shape parameters.
- Modules for visualizing shape observables and their optimization, as in the GIF above.

## Installation

### From PyPI

In your Python environment run

```
python -m pip install pyshaper
# python -m pip install --upgrade 'pyshaper[all]'  # for all extras
```

### From this repository locally

In your Python environment from the top level of this repository run

```
python -m pip install .
# python -m pip install --upgrade '.[all]'  # for all extras
```

### From GitHub

In your Python environment run

```
python -m pip install "pyshaper @ git+https://github.com/rikab/shaper.git"
# python -m pip install --upgrade "pyshaper[all] @ git+https://github.com/rikab/shaper.git"  # for all extras
```

## Example Usage

For an example of how to use `SHAPER`, see the notebook `examples/example.ipynb`. This notebook contains example code for loading data, using pre-built shape observables, defining custom shape observables, running the `SHAPER` algorithm to evaluate these observables, and making plots.

To run the example, you will need to have `pyshaper` installed with all extras. This can be done using (assuming a PyPi installation):

```
python -m pip install --upgrade 'pyshaper[all]'
```

See the Installation section above for more details.

## Dependencies

To use `SHAPER`, the following packages must be installed as prerequisites:

- [PyTorch](https://github.com/pytorch/pytorch): A standard tensor operation library.
- [GeomLoss](https://www.kernel-operations.io/geomloss/): A library for optimal transport.
- [pyjet](https://github.com/scikit-hep/pyjet): A package for jet clustering, needed for default observable definitions.
- [Energyflow](https://energyflow.network/): A suite of particle physics tools. This package is OPTIONAL; however, many of the example datasets within `SHAPER` require this package to load. Not necessary if you provide and format your own data. Included as part of the 'energyflow' extra.
- [imageio](https://pypi.org/project/imageio/): An image manipulation package. Needed for automatic gif creation -- not needed otherwise. Included as part of the 'viz' extra.
- Standard python packages: [numpy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/)

## Citation

If you use `SHAPER`, please cite both this code archive and the corresponding paper, "Can You Hear the Shape of a Jet"?:

    @software{SHAPER,
      author = {Rikab Gambhir, Akshunna S. Dogra, Abiy Tasissa, Demba Ba, Jesse Thaler},
      title = "{pyshaper: v1.1.0}",
      version = {1.1.0},
      doi = {10.5281/zenodo.7689890},
      url = {doi.org/10.5281/zenodo.7689890},
      note = {https://github.com/rikab/SHAPER/releases/tag/v1.1.0}
    }

    @article{Ba:2023hix,
    author = "Ba, Demba and Dogra, Akshunna S. and Gambhir, Rikab and Tasissa, Abiy and Thaler, Jesse",
    title = "{SHAPER: Can You Hear the Shape of a Jet?}",
    eprint = "2302.12266",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "MIT-CTP 5535",
    month = "2",
    year = "2023"
    }

## Changelog

- v1.1.1: 28 June 2023. Pairwise EMDs added.
- v1.1.0: 5 May 2023. Updated Geomloss dependency.
- v1.0.1: 10 March 2023. PyPi-installable release. Minor changes to example and optional dependency handling.
- v1.0.0: 24 February 2023. Official public release.

Based on the work in ["SHAPER: Can You Hear the Shape of a Jet?" (arxiv:2302.12266)](https://arxiv.org/abs/2302.12266)

Bugs, Fixes, Ideas, or Questions? Contact me at rikab@mit.edu

To discuss finer mathematical details (model convergence, optimization guarantees, etc), you may also contact Akshunna S. Dogra at adogra@nyu.edu
