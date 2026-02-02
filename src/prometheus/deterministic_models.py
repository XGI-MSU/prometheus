"""
Holds the class describing all deterministic signals.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional

from .data import Data



class DeterministicModel:

    """
    Class for general deterministic signal models.

    Required Attributes
    -------------------
    name : str
        The name of the deterministic signal.
    data : Data
        An instance of the Data class from the `data` module.
    get_delays_func : Callable
        A JAX friendly function which takes in parameters of the
        deterministic model and outputs the induced timing delays
        across all pulsars. See deterministic.py for example functions.
    parameter_bounds : array
        The minima and maxima allowed values of the parameters,
        with shape (nparams, 2) where nparams is the length of the parameters
        supplied to 'get_delays_func'. The parameter minima are at [:, 0] and
        the maxima at [:, 1].
    get_coeffs_func : Callable
        A JAX friendly function which maps the parameters of the deterministic
        model to the frequency representation of the signal. If None, defaults
        to the FFT method below.
    null : bool
        Whether or not to use the 'null' determinstic model. The null deterministic
        model depends on no parameters and induces no timing delays. This is to avoid
        branching down the pipeline: technically Prometheus always includes a deterministic
        signal. If the user doesn't specify a deterministic signal, use the null model.
    with_psr_params : bool
        Deterministic signals from individual binaries depend on a set of (npsr) pulsar
        phase parameters and (npsr) pulsar distance parameters, where npsr is the number
        of pulsars in the array. If True, the deterministic model will automatically sample
        and supply these parameters to the 'get_delays_func', without the user having to
        specify their bounds and priors. If False, the pipeline will wrap 'get_delays_func'
        to accept these parameters anyway and feed it NoneType in those parameter slots.
    """

    def __init__(self,
                 name : str,
                 data : Data,
                 get_delays_func : Callable,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 get_coeffs_func : Optional[Callable] = None,
                 null : Optional[bool] = False,
                 with_psr_params : Optional[bool] = True):
        self.name = name
        self.data = data
        self.get_delays_func = get_delays_func
        self.parameter_bounds = parameter_bounds
        self.parameter_bounds = jnp.array(parameter_bounds)
        self.param_mins = self.parameter_bounds[:, 0]
        self.param_maxs = self.parameter_bounds[:, 1]
        self.nparams = self.param_mins.shape[0]
        self.null = null
        self.with_psr_params = with_psr_params

        # if pulsar parameters not needed for model, wrap input function
        self.get_delays_func = jax.jit(self._wrap_delays_func(self.get_delays_func,
                                                              self.with_psr_params))

        # if no get_coeffs_func specified, use FFT method
        self.get_coeffs_func = get_coeffs_func or self.get_coeffs_via_FFT
        self.get_coeffs_func = jax.jit(self.get_coeffs_func)

        # frequency bins used in deterministic model
        self.nfreqs = self.data.num_coeff_det // 2
        self.num_coeff_det = self.data.num_coeff_det
        

    def get_coeffs_via_FFT(self, det_params, psr_phases, psr_dists):
        """
        Mapping from deterministic signal parameters to Fourier space.
        This is simply a FFT.
        
        Note the frequency bins used here are generally different than those used
        by stochastic models. To avoid Gibbs phenomena from non-periodic deterministic
        signals over Tspan, we use an "extended" basis where we FFT the signal over a period
        extended either side of Tspan after applying a Tukey window. The Fourier design matrix,
        however, maps the Fourier coefficients to TOAs **within** the PTA Tspan. This introduces
        cross terms in the posterior. See Gundersen & Cornish 2025.
        
        Parameters
        ----------
        det_params : array
            Parmeters values of the deterministic model.
        psr_phases : array or None
            The phase of the gravitational wave at each pulsar. This is a model parameter
            for continuous gravitational waves from individual SMBHBs. For other deterministic
            models, this can be None.
        psr_dists : array or None
            The distance to each pulsar [kpc]. This is a model parameter for continuous
            gravitational waves from individual SMBHBs. For other deterministic models,
            this can be None.

        Returns
        -------
        coeff : array
            A (npsrs, 2*nfreq) array where npsrs is the number of pulsars in the array and
            nfreq is the number of frequency bins used to represent to the deterministic
            model in Fourier space.
        """
        
        # get timing delays induced by the deterministic signal over "sparse" (evenly-spaced) TOAs
        det_residuals = self.get_delays_func(self.data.sparse_toas_det, self.data.psrpos,
                                                    det_params, psr_phases, psr_dists)
        # window residuals over extended observation
        det_residuals_windowed = self.data.Tukey_det * det_residuals
        # do FFT
        det_fft = jnp.fft.fft(det_residuals_windowed, n=None, axis=-1, norm=None)  # dim (Np, 2 * Nf + 2)
        # apply time shift to set initial time
        det_fft *= jnp.exp(-1.j * 2 * jnp.pi * self.data.freqs_forFFT * self.data.sparse_toas_det[:, 0:1])
        
        # extract sine and cosine coefficients
        a_n = jnp.imag(det_fft[:, :self.data.Nsparse // 2]) * (-2 / self.data.Nsparse)  # (Np, Nf + 1)
        b_n = jnp.real(det_fft[:, :self.data.Nsparse // 2]) * (2 / self.data.Nsparse)  # (Np, Nf + 1)

        # interweave sine/cosine coefficients and reshape to (Np, 2 * Nf)
        # coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((self.data.npsrs, 2, self.data.nfreqs + 1))\
        #                         .transpose((0, 2, 1)).reshape((self.data.npsrs, self.data.ncomponents + 2))
        coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((self.data.npsrs, 2, self.nfreqs + 1))\
                        .transpose((0, 2, 1)).reshape((self.data.npsrs, self.data.num_coeff_det + 2))
        return coeff[:, 2:]  # remove DC


    def _wrap_delays_func(self, func, with_psr_params):
        """
        If a deterministic model is supplied that does **not**
        take pulsar distance and phase as model parameters,
        then the 'get_delay_func' is wrapped to accept these
        parameters anyway and the sampler will supply NoneTypes
        for these inputs.
        """
        if with_psr_params:
            return func
        else:
            def wrapped(toas, psrpos, params, psr_phases, psr_dists):
                return func(toas, psrpos, params)
            return wrapped


def build_null_deterministic_model(data):
    """
    Function which builds the 'null' DeterministicModel instance.

    When the user doesn't supply a deterministic model, i.e. they
    only desire a stochastic analysis, the null deterministic model
    is used to avoid branching in the pipepline. When in use,
    the null model doesn't sample any parameters, and the delays
    induced are fixed to zero.
    
    Parameters
    ----------
    data : Data
        An instance of the Data object found in the data.py module.

    Returns
    -------
    null_deterministic_model : DeterministicModel
        An instance of the DeterministicModel class which induces
        zero timing delays.
    """
    def null_det_delays(toas, psrpos, parameter_values, psr_phases, psr_dists):
        return jnp.zeros_like(toas)
    
    def null_get_coeffs_func(parameter_values, psr_phases, psr_dists):
        return jnp.zeros((data.npsrs, data.num_coeff_det))
    
    null_deterministic_model = DeterministicModel(name='null_model',
                                                  data=data,
                                                  get_delays_func=null_det_delays,
                                                  parameter_bounds=jnp.zeros((2, 2)),
                                                  get_coeffs_func=null_get_coeffs_func,
                                                  null=True,
                                                  with_psr_params=False)
    return null_deterministic_model
