"""
Constructs the C-hat covariance matrix via FFT integration. Implements the 
method of arXiv:2506.13866. All PSD values from spectra.py are in [ns]², 
so Ĉ is in [ns]².
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from . import spectra


def build_chat(spectra_func, Nf, T, oversample=3, fmax_factor=1, cutoff=1,
               dtype=jnp.float32):
    """
    Construct the C-hat matrix from a prometheus spectra.py PSD function.

    Parameters
    ----------
    spectra_func : callable
        A *continuous* PSD function from spectra.py
    Nf : int
        Number of canonical Fourier frequency bins (f_k = k/T, k = 1...Nf).
    T : float
        Observation span in seconds.
    oversample : int
        PSD oversampling factor ω; internal frequency step.
        Larger values capture steep/narrow spectra more accurately.
    fmax_factor : int
        Multiplies the maximum sampled frequency by this factor, extending
        the integration range to fmax_factor * Nf/T.
    cutoff : int or None
        Low-frequency cutoff: frequencies below 1/T are zeroed (standard PTA
        convention, matching Discovery's default ``cutoff=1``).  Set to None
        to disable.
    dtype : jax dtype
        Floating-point precision.  Default ``jnp.float32``.

    Returns
    -------
    n_t : int
        Number of coarse time-grid nodes n_t = 2Nf + 1 (odd).
    covmat : callable
        ``covmat(params)`` -> (n_t, n_t) float array.
        Toeplitz c-hat matrix in [ns]², where ``params`` is passed directly to
        ``spectra_func`` as its first argument.
    """
    if spectra_func is spectra.free_spectral:
        raise ValueError(
            "build_chat requires a continuous PSD model "
            "(spectra.power_law or spectra.power_law_flat_tail). "
            "spectra.free_spectral has no continuum interpolation."
        )
    if not all(isinstance(x, int) for x in (Nf, oversample, fmax_factor)):
        raise TypeError("Nf, oversample, and fmax_factor must be integers.")
    if cutoff is not None and not isinstance(cutoff, int):
        raise TypeError("cutoff must be an integer or None.")

    # coarse nodes (odd).
    n_t = 2 * Nf + 1
    scaled_components = (n_t - 1) * fmax_factor + 1

    # Internal oversampled frequency grid
    df_fine = 1.0 / (T * oversample)
    n_max = (scaled_components - 1) // 2 * oversample
    m_all = np.arange(1, n_max + 1)
    fine_freqs = jnp.asarray(m_all * df_fine, dtype=dtype)   # (n_max,)

    # Low-frequency cutoff mask: zero out m < i_cutoff  (f < 1/T).
    if cutoff is not None:
        i_cutoff = int(np.ceil(oversample / cutoff))
        mask = jnp.asarray(m_all >= i_cutoff, dtype=dtype)   # (n_max,)
    else:
        mask = None

    df32 = dtype(df_fine)

    def covmat(params):
        params = jnp.asarray(params, dtype=dtype)

        phi_fine = spectra_func(params, fine_freqs)          # (2·n_max,)
        psd_fine = (phi_fine[0::2] / df32).astype(dtype)    # S(f_m), (n_max,)

        if mask is not None:
            psd_fine = psd_fine * mask

        psd = jnp.concatenate([jnp.zeros(1, dtype=dtype), psd_fine])  # (n_max+1,)
        fullpsd = jnp.concatenate((psd, psd[-2:0:-1]))                 # (2·n_max,)
        Cfreq = jnp.fft.ifft(fullpsd)
        Ctau = (Cfreq.real * len(fullpsd) * df32 / 2).astype(dtype)   # (2·n_max,)
        return jsl.toeplitz(Ctau[:scaled_components:fmax_factor])      # (n_t, n_t)

    return n_t, covmat


def get_coarse_times(T, Nf, t0=0.0):
    """
    Coarse time-grid nodes for the C-hat matrix (N-hat = 2*Nf + 1 points, [t0, t0+T]).

    Parameters
    ----------
    T   : float   Observation span in seconds.
    Nf  : int     Number of frequency bins.
    t0  : float   Start time (default 0).

    Returns
    -------
    times : ndarray, shape (2·Nf + 1,)
    """
    N_hat = 2 * Nf + 1
    return t0 + np.linspace(0.0, T, N_hat)


def _fourier_design(n_t, T, Nf, dtype=np.float64):
    """
    Real Fourier design matrix on an n_t-point grid spanning [0, T].

    Using integer lag indices avoids float precision loss that
    would occur if an absolute t0 offset were included.

    Returns
    -------
    F : ndarray, shape (n_t, 2·Nf), dtype float64 (cast externally as needed)
    """
    dt = T / n_t
    lags = np.arange(n_t)                          # [0, 1, …, n_t-1]
    freqs = np.arange(1, Nf + 1) / T               # [1/T, 2/T, …, Nf/T]
    phase = 2.0 * np.pi * dt * np.outer(lags, freqs)  # (n_t, Nf)
    F = np.empty((n_t, 2 * Nf), dtype=dtype)
    F[:, 0::2] = np.sin(phase)
    F[:, 1::2] = np.cos(phase)
    return F


def build_phi_from_chat(spectra_func, dataset, oversample=8, jitter_scale=10.0,
                        dtype=jnp.float32):
    """
    Build the (2Nf, 2Nf) frequency-domain covariance matrix phi from the
    time-domain C-hat matrix.

    Parameters
    ----------
    spectra_func : callable
        A *continuous* PSD function from ``spectra.py``.
        Valid for ``spectra.power_law`` and ``spectra.power_law_flat_tail``.
        NOT valid for ``spectra.free_spectral``.
    dataset : prometheus.data.Data
        Prometheus data object.
    oversample : int
        Number of oversampled frequency bins per canonical bin.
    jitter_scale : float
        Scales the diagonal regularisation jitter. Default 10.
        Set to 0 to disable.
    dtype : jax dtype
        Floating-point precision.  Default ``jnp.float32``.

    Returns
    -------
    get_phi : callable
        ``get_phi(params)`` → (2Nf, 2Nf) float array in [ns]².
        Frequency-domain covariance matrix with full inter-frequency
        correlations, positive definite (up to jitter), symmetric.
    """
    if spectra_func is spectra.free_spectral:
        raise ValueError(
            "build_phi_from_chat requires a continuous PSD model "
            "(spectra.power_law or spectra.power_law_flat_tail). "
            "spectra.free_spectral has no continuum interpolation."
        )

    Nf = dataset.nfreqs
    T = float(dataset.Tspan)
    n_t = 2 * Nf + 1

    n_os = Nf * oversample
    df_os = 1.0 / (T * oversample)
    m_all = np.arange(1, n_os + 1)
    f_os = jnp.asarray(m_all * df_os, dtype=dtype)

    # Low-frequency cutoff: zero out m < oversample  ↔  f < 1/T.
    mask_os = jnp.asarray(m_all >= oversample, dtype=dtype)

    # Coarse time-lag indices (integer, avoids large-phase trig error).
    dt_coarse = dtype(T / n_t)
    lag_ints = jnp.arange(n_t, dtype=dtype)

    # Fourier design matrix F on the coarse grid
    F_coarse = jnp.asarray(_fourier_design(n_t, T, Nf), dtype=dtype)

    # Normalisation
    norm = dtype(4.0 / n_t ** 2)
    eps = jnp.finfo(dtype).eps

    def get_phi(params):
        params = jnp.asarray(params, dtype=dtype)
        phi_os = spectra_func(params, f_os)[0::2].astype(dtype) * mask_os
        phase = (dtype(2.0 * np.pi) * dt_coarse) * (lag_ints[:, None] * f_os[None, :])
        Cos = jnp.cos(phase)
        Sin = jnp.sin(phase)

        with jax.default_matmul_precision('highest'):

            FC = F_coarse.T @ Cos      # (2Nf, n_os)
            FS = F_coarse.T @ Sin      # (2Nf, n_os)
            phi = norm * ((FC * phi_os[None, :]) @ FC.T + (FS * phi_os[None, :]) @ FS.T)

        phi = 0.5 * (phi + phi.T)
        jitter = jitter_scale * eps * jnp.max(jnp.diag(phi))
        return (phi + jitter * jnp.eye(2 * Nf, dtype=dtype)).astype(dtype)

    return get_phi

