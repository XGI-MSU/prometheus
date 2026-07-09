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
    log10_phi_diag_power_law = jnp.log10(power_law(jnp.array([log10_A, gamma]), freqs))
        
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



def make_linear_spline_free_spectral_model(spline_freqs):
    """
    Construct a spline-based free spectral model.

    Parameters
    ----------
    spline_freqs : array
        Frequencies at which the free spectral parameters are defined.
        Shape (Nk,).

    Returns
    -------
    spectral_model : callable
        Function with signature
            phi_diag = spectral_model(spectral_parameters, freqs)
        where spectral_parameters are log10(rho) values at spline_freqs.
    """

    log10_spline_freqs = jnp.log10(spline_freqs)

    def spectral_model(spectral_parameters, freqs):

        log10_freqs = jnp.log10(freqs)

        # interior interpolation
        log10_rho_interp = jnp.interp(
            log10_freqs,
            log10_spline_freqs,
            spectral_parameters,
            left=spectral_parameters[0],
            right=spectral_parameters[-1],
        )

        log10_rho = log10_rho_interp

        # repeat for sine/cosine basis
        log10_rho_repeated = jnp.repeat(log10_rho, 2)

        # convert to covariance units [ns]^2
        phi_diag = 10 ** (
            log10_rho_repeated
            + 2 * utils.log10_renorm
        )

        return phi_diag

    return spectral_model



def make_cubic_spline_free_spectral_model(spline_freqs):

    x = jnp.log10(spline_freqs)
    n = len(x)

    def spectral_model(spectral_parameters, freqs):

        y = spectral_parameters

        #
        # Natural cubic spline:
        # solve for second derivatives M
        #

        h = x[1:] - x[:-1]


        lower = h[:-1]
        diag = 2.0 * (h[:-1] + h[1:])
        upper = h[1:]

        rhs = 6.0 * (
            (y[2:] - y[1:-1]) / h[1:]
            - (y[1:-1] - y[:-2]) / h[:-1]
        )

        A = (
            jnp.diag(diag)
            + jnp.diag(upper[:-1], 1)
            + jnp.diag(lower[1:], -1)
        )

        M_inner = jnp.linalg.solve(A, rhs)

        M = jnp.concatenate(
            [
                jnp.zeros(1, dtype=y.dtype),
                M_inner,
                jnp.zeros(1, dtype=y.dtype),
            ]
        )

        #
        # Cubic spline interpolation
        #

        x_eval = jnp.log10(freqs)

        idx = jnp.searchsorted(x, x_eval, side="right") - 1
        idx = jnp.clip(idx, 0, n - 2)

        x_i = x[idx]
        x_ip1 = x[idx + 1]

        y_i = y[idx]
        y_ip1 = y[idx + 1]

        M_i = M[idx]
        M_ip1 = M[idx + 1]

        h_i = x_ip1 - x_i

        a = (x_ip1 - x_eval) / h_i
        b = (x_eval - x_i) / h_i

        spline_val = (
            a * y_i
            + b * y_ip1
            + ((a**3 - a) * M_i * h_i**2) / 6.0
            + ((b**3 - b) * M_ip1 * h_i**2) / 6.0
        )

        log10_rho = jnp.where(
            x_eval < x[0],
            spectral_parameters[0],
            jnp.where(
                x_eval > x[-1],
                spectral_parameters[-1],
                spline_val,
            ),
        )

        log10_rho_repeated = jnp.repeat(log10_rho, 2)

        phi_diag = 10 ** (
            log10_rho_repeated
            + 2 * utils.log10_renorm
        )

        return phi_diag

    return spectral_model



def make_akima_spline_free_spectral_model(spline_freqs, extrapolate="constant"):
    """
    Construct an Akima-spline-based free spectral model.

    Parameters
    ----------
    spline_freqs : array
        Frequencies at which the free spectral parameters are defined.
        Shape (Nk,), Nk >= 3.
    extrapolate : str, optional
        How to evaluate the spline outside the range of ``spline_freqs``.
        "constant" (default) holds the value fixed at the first/last
        spectral_parameters, matching the behavior of
        make_cubic_spline_free_spectral_model. "akima" instead
        extrapolates using the cubic Hermite polynomial of the
        boundary segment of the Akima spline.

    Returns
    -------
    spectral_model : callable
        Function with signature
            phi_diag = spectral_model(spectral_parameters, freqs)
        where spectral_parameters are log10(rho) values at spline_freqs.
    """

    if extrapolate not in ("constant", "akima"):
        raise ValueError('extrapolate must be "constant" or "akima"')

    x = jnp.log10(spline_freqs)
    n = len(x)

    def spectral_model(spectral_parameters, freqs):

        y = spectral_parameters

        #
        # Akima spline:
        # estimate derivatives at each knot from
        # locally-extended secant slopes
        #

        h = x[1:] - x[:-1]
        m = (y[1:] - y[:-1]) / h

        m_left1 = 2.0 * m[0] - m[1]
        m_left2 = 2.0 * m_left1 - m[0]
        m_right1 = 2.0 * m[-1] - m[-2]
        m_right2 = 2.0 * m_right1 - m[-1]

        m_ext = jnp.concatenate(
            [
                jnp.array([m_left2, m_left1]),
                m,
                jnp.array([m_right1, m_right2]),
            ]
        )

        dm = jnp.abs(m_ext[1:] - m_ext[:-1])
        f1 = dm[2 : 2 + n]
        f2 = dm[0:n]
        denom = f1 + f2

        t = jnp.where(
            denom > 1e-12,
            (f1 * m_ext[1 : 1 + n] + f2 * m_ext[2 : 2 + n]) / jnp.where(denom > 1e-12, denom, 1.0),
            0.5 * (m_ext[1 : 1 + n] + m_ext[2 : 2 + n]),
        )

        #
        # Akima (Hermite) spline interpolation
        #

        x_eval = jnp.log10(freqs)

        idx = jnp.searchsorted(x, x_eval, side="right") - 1
        idx = jnp.clip(idx, 0, n - 2)

        x_i = x[idx]
        x_ip1 = x[idx + 1]

        y_i = y[idx]
        y_ip1 = y[idx + 1]

        t_i = t[idx]
        t_ip1 = t[idx + 1]

        h_i = x_ip1 - x_i
        s = (x_eval - x_i) / h_i

        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2

        spline_val = (
            h00 * y_i
            + h10 * h_i * t_i
            + h01 * y_ip1
            + h11 * h_i * t_ip1
        )

        if extrapolate == "constant":
            log10_rho = jnp.where(
                x_eval < x[0],
                spectral_parameters[0],
                jnp.where(
                    x_eval > x[-1],
                    spectral_parameters[-1],
                    spline_val,
                ),
            )
        else:
            log10_rho = spline_val

        #
        # Convert to PTA covariance convention
        #

        log10_rho_repeated = jnp.repeat(log10_rho, 2)

        phi_diag = 10 ** (
            log10_rho_repeated
            + 2 * utils.log10_renorm
        )

        return phi_diag

    return spectral_model

