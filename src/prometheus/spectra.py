"""
Functions to help build instances of the spectral model objects
found in spectral_models.py. For example, a function which maps
power law parameters (log10_A, gamma) to the diagonal of the 
prior covariance matrix, phi, is found here.
"""


import jax.numpy as jnp
from . import utilities as utils


def power_law(spectral_parameters, freqs):
    """
    Get the diagonal of the prior covariance matrix for a 
    Gaussian process obeying a power law.
    
    Parameters
    ----------
    spectral_parameters : array
        Array of power law spectral parameters in order
        log10(amplitude), spectral index.
    freqs : array
        Array of frequencies of shape (Nf,) where Nf is
        the number of frequency bins.

    Returns
    -------
    phi_vecs : array
        Array of shape (2*Nf,) where Nf is the number of frequency
        bins. The array is the diagonal of the (prior) covariance matrix
        for the Fourier coefficients obeying a power law.
        We use base units of [ns], so the elements of the
        covariance matrix output have units of [ns]^2.
    """
    # unpack power law parameters
    log10_A, gamma = spectral_parameters

    # frequency array with zero frequency
    df = jnp.diff(jnp.concatenate((jnp.array([0]), freqs)))

    # power law spectrum
    log10_phi_diags = 2 * utils.log10_renorm + 2 * log10_A - jnp.log10(12.0 * jnp.pi**2) \
        + (gamma - 3)*jnp.log10(utils.fyr) + (-gamma)*jnp.log10(jnp.repeat(freqs, 2)) \
            + jnp.log10(jnp.repeat(df, 2))
    phi_diags = 10**log10_phi_diags
    return phi_diags


def power_law_flat_tail(spectral_parameters, freqs):
    """
    Get the diagonal of the covariance matrix for a 
    Gaussian process obeying a power law with a flat tail.
    
    Parameters
    ----------
    spectral_parameters : array
        Array of power law with flat tail spectral parameters
        in order log10(amplitude), spectral index, log10(amplitude of tail).
    freqs : array
        Array of frequencies of shape (Nf,) where Nf is
        the number of frequency bins.

    Returns
    -------
    phi_vecs : array
        Array of shape (2*Nf,) where Nf is the number of frequency
        bins. The array is the diagonal of the (prior) covariance matrix
        for the Fourier coefficients obeying a power law with a flat tail.
        We use base units of [ns], so the elements of the
        covariance matrix output have units of [ns]^2.
    """
    # unpack power law with flat tail parameters
    log10_A, gamma, log10_kappa = spectral_parameters

    # power law spectrum
    log10_phi_diag_power_law = power_law(jnp.array([log10_A, gamma]), freqs)
    
    # flat tail spectrum
    log10_phi_diag_flat = 2 * utils.log10_renorm + 2 * log10_kappa

    # power law when above tail, otherwise flat tail distribution
    log10_phi_diag = jnp.maximum(log10_phi_diag_power_law, log10_phi_diag_flat)
    
    phi_diag = 10**log10_phi_diag

    return phi_diag


def free_spectral(spectral_parameters, freqs):
    """
    Get the diagonal of the covariance matrix for a 
    Gaussian process obeying a free spectral model.
    
    Parameters
    ----------
    spectral_parameters : array
        Array of free spectral parameters
        in order log10(rho_1), log10(rho_2), log10(rho_2), ...
        where rho_i is the free power in the ith frequency bin.
    freqs : array
        Array of frequencies of shape (Nf,) where Nf is
        the number of frequency bins.

    Returns
    -------
    phi_vecs : array
        Array of shape (2*Nf,) where Nf is the number of frequency
        bins. The array is the diagonal of the (prior) covariance matrix
        for the Fourier coefficients obeying a free spectral model.
        We use base units of [ns], so the elements of the
        covariance matrix output have units of [ns]^2.
    """
    # log-power in each frequency bin are parameters
    log10_rho = spectral_parameters

    # repeat power in each frequency for sine/cosine basis
    log10_rho_repeated = jnp.repeat(log10_rho, 2)

    # if less free spectral parameters are supplied than
    # frequency bins, then append zeros
    # (this might be necessary when a GWB is modeled with fewer
    # frequency bins than the pulsar noise)
    zeros = jnp.zeros(2 * freqs.shape[0] - log10_rho_repeated.shape[0])
    log10_rho_extended = jnp.concatenate((log10_rho_repeated, zeros))

    # convert to units of [ns]^2
    phi_diag = 10**(log10_rho_extended + 2 * utils.log10_renorm)

    return phi_diag

