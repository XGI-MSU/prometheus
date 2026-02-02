"""
Holds classes for stochastic Gaussian spectral models.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional

from . import utilities as utils
from .data import Data


class SpectralModel:
    """
    Base class for spectral models corresponding to Gaussian processes.
    An advanced user can use this class for their stochastic models instead
    of the models below, which are limited (but easier to implement).
    
    Required Attributes
    -------------------
    name : str
        The name of the spectral model.
    parameter_bounds : array
        (Nparam, 2) array where 'Nparam' is the total
        number of parameters of the spectral model.
        The ordering is such that index [:, 0] gives the
        minima of parameters and [:, 1] the maxima.
    data : data.Data
        An instance of the Data class from the `data` module.
    get_phi_cube_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function which takes an array of parameter values of shape (Nparam,)
        where 'Nparam' is the total number of parameters of the spectral model and
        an array of frequencies of shape (2*Nf,) where 'Nf' is the number of
        unique frequency bins. The frequency array should be repeated for the two
        bases- sine and cosine, e.g. [1/Tspan, 1/Tspan, 2/Tspan, 2/Tspan, ...].
        The function should output a (2*Nf, Npsrs, Npsrs) array where 'Npsrs' is the
        number of pulsars in the array. The (i, j, k) element of the output array
        is the (prior) covariance of ith Fourier coefficient in pulsars j and k.
        The output should have units of [ns]^2.

        **See the advanced modeling example notebooks for examples of get_phi_cube_func.**
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 get_phi_cube_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 ):
        
        self.name = name
        self.parameter_bounds = jnp.array(parameter_bounds)
        self.param_mins = self.parameter_bounds[..., 0]
        self.param_maxs = self.parameter_bounds[..., 1]
        self.nparams_base = self.param_mins.shape[0]
        self.data = data

        if get_phi_cube_func is None:
            raise ValueError("get_phi_cube_func must be provided.")

        self._get_phi_cube_func = get_phi_cube_func

    def get_phi_cube(self, params, freqs):
        """
        Gets the prior covariance matrix for the Fourier coefficients
        conditioned on this spectral model and some parameters.

        Parameters
        ----------
        params : array
            Array of spectral-hyper-parameters.
        freqs : array
            Array of shape (2*Nf,) of frequencies, where Nf is the number
            of frequency bins modeled. Each frequency should be repeated
            for the sine/cosine basis, e.g. [1/Tspan, 1/Tspan, 2/Tspan, 2/Tspan,...].
        
        Returns
        -------
        phi_cube : array
            Array of shape (2*Nf, Npsrs, Npsrs) where Nf is the number of frequency
            bins modeled and Npsrs is the number of pulsars in the array. This 
            array is the prior covariance of the Fourier coefficients such that
            the (i, j, k) element is the covariance of the ith Fourier coefficient
            between pulsars j and k. The output array should have units of [ns]^2.
        """
        return self._get_phi_cube_func(params, freqs)


class IndependentSpectralModel(SpectralModel):

    """
    Spectral model which vectorizes across pulsars. Used primarily for 
    pulsar noise models, where each pulsar uses the same (statistically
    independent) model (e.g. power law).

    Inherits from :class:`SpectralModel`.

    Required Attributes
    -------------------
    name : str
        The name of spectral model.
    parameter_bounds : array
        The minima and maxima allowed values of the parameters
        The shape of the array is (Nparam, 2) where 'Nparam'
        is number of parameters of the spectral model (per pulsar).
        The ordering is such that index [:, 0] gives the
        minima of parameters and [:, 1] the maxima.
    data : data.Data
        An instance of the Data class from the `data` module.
    get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function which takes an array of parameter values of shape (Nparam,)
        where 'Nparam' is the number of parameters of the spectral model and
        an array of frequencies of shape (2*Nf,) where 'Nf' is the number of
        unique frequency bins. The frequency array should be repeated for the two
        bases- sine and cosine, e.g. [1/Tspan, 1/Tspan, 2/Tspan, 2/Tspan, ...].
        The function outputs an array of shape (2*Nf,) which is the diagonal
        elements of prior covariance matrix for the Fourier coefficients obeying
        that spectral model. The output array must should use units of [ns]^2.
        See spectra.py for examples.
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 ):
        
        self.get_phi_diag_func = get_phi_diag_func

        # vectorize spectral model over all pulsars
        # so each pulsar gets an identical independent model
        vectorized_diag = jax.vmap(
            get_phi_diag_func,
            in_axes=(0, None),
        )

        def get_phi_cube_func(params, freqs):
            """
            Gets the prior covariance matrix for the Fourier coefficients
            conditioned on this spectral model and some parameters.

            Parameters
            ----------
            params : array
                Array of spectral-hyper-parameters.
            freqs : array
                Array of shape (2*Nf,) of frequencies, where Nf is the number
                of frequency bins modeled. Each frequency should be repeated
                for the sine/cosine basis, e.g. [1/Tspan, 1/Tspan, 2/Tspan, 2/Tspan,...].
            
            Returns
            -------
            phi_cube : array
                Array of shape (2*Nf, Npsrs, Npsrs) where Nf is the number of frequency
                bins modeled and Npsrs is the number of pulsars in the array. This 
                array is the prior covariance of the Fourier coefficients such that
                the (i, j, k) element is the covariance of the ith Fourier coefficient
                between pulsars j and k. The covariance should use units of [ns]^2.
            """
            phi_diags = vectorized_diag(params, freqs).T
            nfreqs2, npsrs = phi_diags.shape
            phi_cube = jnp.zeros((nfreqs2, npsrs, npsrs))
            ii = jnp.arange(npsrs)
            return phi_cube.at[:, ii, ii].set(phi_diags)

        super().__init__(
            name=name,
            parameter_bounds=parameter_bounds,
            data=data,
            get_phi_cube_func=get_phi_cube_func,
        )


class CommonSpectralModel(SpectralModel):

    """
    Spectral model which implements a common process over pulsar under
    some correlation pattern. Used primarily for stochastic gravitational
    wave background models.

    Inherits from :class:`SpectralModel`.

    Required Attributes
    -------------------
    name : str
        The name of spectral model.
    parameter_bounds : list
        The minima and maxima allowed values of the parameters
        The shape of the array is (Nparam, 2) where 'Nparam'
        is number of parameters of the spectral model.
        The ordering is such that index [:, 0] gives the
        minima of parameters and [:, 1] the maxima.
    data : data.Data
        An instance of the Data class from the `data` module.
    get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function which takes an array of parameter values of shape (Nparam,)
        where 'Nparam' is the number of parameters of the spectral model and
        an array of frequencies of shape (2*Nf,) where 'Nf' is the number of
        unique frequency bins. The frequency array should be repeated for the two
        bases- sine and cosine, e.g. [1/Tspan, 1/Tspan, 2/Tspan, 2/Tspan, ...].
        The function outputs an array of shape (2*Nf,) which is the diagonal
        elements of prior covariance matrix for the Fourier coefficients obeying that
        spectral model. The covariance matrix should use units of [ns]^2.
        See spectra.py for examples.
    correlation_matrix : str or array
        A (Npsrs, Npsrs) array where Npsrs is the number of pulsars in the PTA.
        The array is the correlation pattern between pulsars, e.g. element (i, j)
        is the correlation between pulsars i and j in the array. User may input
        'HD' or 'CURN' for Hellings-Downs and common uncorrelated processes, or
        may supply their own correlation matrix.
    nfreqs : int
        Number of frequencies to use in this spectral model. This value must be
        less than or equal to data.nfreqs. If None, defaults to data.nfreqs.
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 correlation_matrix : str | np.ndarray | jnp.ndarray,
                 nfreqs : Optional[int] = None):
        self.data = data
        self.get_phi_diag_func = get_phi_diag_func
        self.nfreqs = nfreqs or data.nfreqs

        # zeros to append to spectrum if user requests
        # fewer frequency bins than in data object
        self.zeros = jnp.zeros(2 * (self.data.nfreqs - self.nfreqs))
        self.num_coeffs = 2 * self.nfreqs

        # if 'HD' or 'CURN', build correlation matrix for user
        if isinstance(correlation_matrix, str):
            correlation_matrix = utils.resolve_psr_corr_matrix(
                correlation_matrix, data
            )
        self.correlation_matrix = jnp.array(correlation_matrix)

        def get_phi_cube_func(params, freqs):
            """
            Gets the prior covariance matrix for the Fourier coefficients
            conditioned on this spectral model and some parameters.

            Parameters
            ----------
            params : array
                Array of spectral-hyper-parameters.
            freqs : array
                Array of shape (2*Nf,) of frequencies, where Nf is the number
                of frequency bins modeled. Each frequency should be repeated
                for the sine/cosine basis, e.g. [1/Tspan, 1/Tspan, 2/Tspan, 2/Tspan,...].

            Returns
            -------
            phi_cube : array
                Array of shape (2*Nf, Npsrs, Npsrs) where Nf is the number of frequency
                bins modeled and Npsrs is the number of pulsars in the array. This 
                array is the prior covariance of the Fourier coefficients such that
                the (i, j, k) element is the covariance of the ith Fourier coefficient
                between pulsars j and k. The covariance should use units of [ns]^2.
            """
            phi_diag = get_phi_diag_func(params, freqs)
            phi_diag = jnp.concatenate((phi_diag[:self.num_coeffs], self.zeros))
            phi_cube = phi_diag[:, None, None] * correlation_matrix[None, :, :]
            return phi_cube

        super().__init__(
            name=name,
            parameter_bounds=parameter_bounds,
            data=data,
            get_phi_cube_func=get_phi_cube_func,
        )

