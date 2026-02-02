# Prometheus

_Prometheus_ is a package for fast analysis of pulsar timing array datasets. It's not as modular as [Enterprise](https://github.com/nanograv/enterprise) or [Discovery](https://github.com/nanograv/discovery), but supports common models for intrinsic pulsar noise and a stochastic gravitational wave background. Moreover, deterministic contributions are supported, so fast fully joint analyses may be performed.

The posterior density sampled in $\texttt{Prometheus}$ is identical to that of $\texttt{Enterprise}$. However, rather than analytically marginalizing over Fourier coefficients which represent stochastic processes, the coefficients are sampled numerically. This avoids costly matrix inversions resulting in a hyper-effecient posterior evaluation. The posterior is implemented in [JAX](https://jax.readthedocs.io/en/latest/) and supports `jax.jit` and automatic differentiation methods. The sampling is performed using `NumPyro`'s No U-Turn Sampler (NUTS).

This software should be executed on an NVIDIA GPU. On a NVIDIA GeForce RTX 3090, the NANOGrav 15-year stochastic analysis results can be obtained after ~15 minutes. The NANOGrav 15-year continous wave parameter estimation results can be obtained in similar time!

## Requirements / Conventions

$\texttt{Prometheus}$ requires a modern Python with standard packages including `enterprise`. The analysis should be executed on a NVIDIA GPU with [CUDA-enabled JAX](https://docs.jax.dev/en/latest/installation.html).

By default, single precision (`float32`) is used for efficiency. To accomodate this, timing residuals use units of _nano-seconds_. All custom functions (e.g. spectral models) supplied by the user and called in the posterior need to be stable in single precision, JAX compatible, and use base units of nano-seconds. For example, a function which outputs elements of a covariance matrix for stochastic timing contributions must output in units of $[\text{ns}]^2$. See functions in `spectra.py` and `deterministic.py` for examples.

If desired, double precision (`float64`) can be used by modifying `__init__.py`, but this slows down the analysis.


## Examples

The `examples` folder contains a variety of example analyses in interactive Python notebooks.

- `gwb_pe.ipynb` reproduces the NANOGrav 15-year stochastic analysis.

- `cw_pe.ipynb` reproduces the NANOGrav 15-year continuous wave analysis.

- The `advanced_modeling` folder contains examples illustrating the construction and sampling of more complicated models.

## Data structure (`data.py`)

$\texttt{Prometheus}$ stores PTA data in a custom `prometheus.data.Data` object. This object may be constructed from a list of `enterprise.pulsar.PintPulsar` or `FeatherPulsar` objects and a standard white noise dictionary.

In addition to common PTA attributes (number of pulsars, sky locations, etc.), the `Data` object also stores constants used for the evaluation of the posterior (e.g. "TNT", "TNr", etc.).

## Signal and noise models

 - The white noise model is fixed throughout the analysis.
 
 - Linear deviations to the pulsar timing model are analytically marginalized.

### Pulsar noise models (`spectra.py` and `spectral_models.py`)

- Timing delays due to intrinsic pulsar noise are represented with a set of Fourier coefficients.

- These coefficients are modeled (in the prior) as a Gaussian process of zero-mean and covariance that can be parameterized with a wide range of spectral models.

- Currently, inter-frequency correlations are not supported. That is, the prior covariance matrix per-pulsar must be diagonal in frequency-space.

- The puslar noise model is usually represented with an instantiation of a `prometheus.spectral_models.IndependentSpectralModel` object.


### GWB models (`spectra.py` and `spectral_models.py`)

- Timing delays due to a stochastic gravitational wave background are represented with a set of Fourier coefficients. In the current version, these are the same coefficients used to represent the pulsar noise.

- These coefficients are modeled as a Gaussian process of zero-mean and covariance that can be parameterized with spectral models.

- As opposed to the pulsar noise model, the GWB model assumes the spectrum is common among all pulsars and is consistent with some inter-pulsar correlation pattern.

- Currently, inter-frequency correlations are not supported. That is, the prior covariance matrix per-pulsar must be diagonal in frequency-space. The GWB model, of course, allows for inter-pulsar correlations.

- The GWB model is usually represented with an instantiation of a `prometheus.spectral_models.CommonSpectralModel` object.

### Custom Spectral Models (`spectra.py` and `spectral_models.py`)

- For advanced modeling, Gaussian processes may be built with the `prometheus.spectral_models.SpectralModel` object.

- Such objects model the Fourier coefficients as a zero-mean Gaussian process where the covariance can be _nearly_ arbitrarily parameterized.

- As above, inter-frequency correlations are not yet supported. That is, all correlation matrices must be represented as $(2N_f, N_p, N_p)$ arrays, where $N_f$ is the number of frequency bins modeled and $N_p$ is the number of pulsars in the array.


### Deterministic Models (`deterministic.py` and `deterministic_models.py`)

- Arbitrary deterministic models are architecturally supported, provided a function which maps the parameters to the induced timing delays across pulsars. See `deterministic.py` for example functions.

- Deterministic signals are represented with an instantiation of a `prometheus.deterministic_models.DeterministicModel` object.

- The parameters of the deterministic model are directly sampled in the MCMC. However, under the hood, deterministic signals are represented in a Fourier basis.

- Arbitrary deterministic models are "supported", but satisfactory sampling for every model cannot be guaranteed. Chain convergence and sampling diagnostics should always be checked!

- See the `tests` folder where `float32` stability and the Fourier representation of an example deterministic signal is demonstrated.


## PTA Models (`pta_model.py`)

- Constituent spectral and deterministic models may be combined into an instance of `prometheus.pta_model.PTAModel` which constructs the joint posterior.

- The `PTAModel` also provides a `NumPyro` compatible probabilistic sampling model.

- The `PTAModel` operates in two modes: "standard" and "custom". In standard mode both an `IndependentSpectralModel` and `CommonSpectralModel` are required. In custom mode, only a `SpectralModel` is accepted.

