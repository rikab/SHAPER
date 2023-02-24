# SHAPER
## v1.0.0

![3-point-ellipsiness-plus-pileup_event_0](https://user-images.githubusercontent.com/78619093/221254441-36b3bcc4-65fc-4211-aaef-2332c5dd893e.gif)

Pictured: Example of a custom jet algorithm, "3-(Ellipse+Point)iness+Pileup", as evaluated on an example top jet, as the `SHAPER` algorithm computes the value of the observable and optimal parameters.



Python implementation of the `SHAPER` algorithm for computing arbitrary shape observables for collider physics, as defined in arxiv:2302:XXXX. The `SHAPER` algorithm contains modules for defining shape observables using parameterized manifolds, and for using the Sinkhorn divergence to evaluate these observables on data.


## Example Usage

For an example of how to use `SHAPER`, see the notebook `example.ipynb`. This notebook contains example code for loading data, using pre-built shape observables, defining custom shape observables, running the `SHAPER` algorithm to evaluate these observables, and making plots.

## Changelog

* v1.0.0: 24 February 2023. Official public release.

Bugs, Fixes, or Questions? Contact me at rikab@mit.edu
