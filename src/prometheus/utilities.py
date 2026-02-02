'''Store constants and utility functions.'''

import numpy as np
import jax.numpy as jnp
from pyarrow import feather
import pandas as pd


# Constants---------------------------------------------------------------------

# Re-scale to use base unit of [ns] to avoid numerical over/under flow in float32
renorm = 1e9
log10_renorm = jnp.log10(renorm)

# rescales TOA axis for deterministic CW signals
cw_renorm = 1e-10

# times and frequencies
day = 86400.0   # seconds
year = 365.2526 * day   # seconds
fyr = 1 / year    # Hertz
year_months = 12.
year_days = 365.25
us_sec = 1.e-6

# reference time for CW model
tref = 4579200000.  # 53000 * day = 53000 * 86400

# physical constants
c = 299792458.0
G = 6.6743e-11
Msun = 1.9891e30
Tsun = Msun * G / c**3.
kpc = 3.085677581491367e+19
Mpc = 1.e3 * kpc
Tkpc = kpc / c

# log_e(10)
ln10 = jnp.log(10.)


# Utility functions-------------------------------------------------------------
def convert_value(value):
    """
    Convert NumPy arrays to native objects in a way that preserves shape.
    Scalars are converted to native Python types.
    Arrays with ndim > 1 are stored as a dictionary with 'data' and 'shape'.
    """
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            return {"data": value.flatten().tolist(), "shape": value.shape}
        else:
            return value.tolist()
    elif isinstance(value, np.generic):
        # Convert NumPy scalar to Python scalar.
        return value.item()
    else:
        return value


def restore_value(x):
    """
    Restore a value that was saved by convert_value().
    If x is a dict with keys 'data' and 'shape', rebuild a NumPy array with that shape.
    If x is a list, convert it to a NumPy array.
    Otherwise, return x unchanged.
    """
    if isinstance(x, dict) and "data" in x and "shape" in x:
        arr = np.array(x["data"])
        try:
            return arr.reshape([int(el) for el in x["shape"]])
        except Exception as e:
            # In case the reshape fails, return the flat array.
            return arr
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x


def save_chain(samples_dict, filepath='samples.feather', save_coeff_samples=False):
    """
    Save samples from NumPyro chain in feather file.

    Parameters
    ----------
    samples_dict : dict
        The dictionary of chain samples from NumPyro
    filepath : str
        Local destination path to save feather file
    save_coeff_samples : bool
        If True, the (many) Fourier coefficients and whitened Fourier
        coefficients samples are saved. By default False.

    Returns
    -------
    None
    """

    keys_to_skip = set()
    if not save_coeff_samples:
        keys_to_skip = {'a', 'xi'}

    flattened_dict = {}
    for key, val in samples_dict.items():
        if key in keys_to_skip:
            continue
        flattened_dict[key] = convert_value(np.array(val))

    feather.write_feather(pd.DataFrame([flattened_dict]), filepath)
    print(f'Saved chain samples to {filepath}.')



def load_chain(filepath):
    """
    Load MCMC samples from feather file.

    Parameters
    ----------
    filepath : str
        Local destination path to save feather file

    Returns
    -------
    samples : dict
        Dictionary of parameter samples from chain
    """

    # load data frame
    df_loaded = feather.read_feather(filepath)

    # First convert the DataFrame to a list of row dictionaries.
    records = df_loaded.to_dict(orient='records')[0]
    
    # Reconstruct the original dictionary.
    samples = {}
    for key, value in records.items():
        samples[key] = restore_value(value)
    return samples


def phitheta_to_psrpos(phi, theta):
    """
    Convert spherical polar angle sky locations to Cartesian
    unit vectors

    Parameters
    ----------
    phi : array
        Azimuthal angles of pulsars' sky location.
    theta : array
        Polar angles of pulsars' sky location.

    Returns
    -------
    Cart_vecs : array
        Sky locations of pulsars in Cartesian coordinates.
    """
    Cart_vecs = np.array([np.cos(phi)*np.sin(theta),
                          np.sin(phi)*np.sin(theta),
                          np.cos(theta)]).T
    return Cart_vecs


def hdcorrmat(psrpos):
    """
    Get Hellings-Downs (HD) correlation matrix given
    pulsar positions in Cartesian coordinates.

    Parameters
    ----------
    psrpos : array
        Sky locations of pulsars in Cartesian coordinates.

    Returns
    -------
    hdmat : array
        (Npulsar x Npulsar) HD covariance matrix
    """

    cosgamma = np.clip(np.dot(psrpos, psrpos.T), -1, 1)
    npsrs = len(cosgamma)

    xp = 0.5 * (1 - cosgamma)

    # The settings make numpy ignore warnings due to numerical precision
    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old_settings)

    hdmat = logxp - 0.25 * xp + 0.5 + 0.5 * np.diag(np.ones(npsrs))

    return hdmat


def resolve_psr_corr_matrix(correlation_matrix_name, data):
    """
    Return common pulsar correlation matrices.
    
    Parameters
    ----------
    correlation_matrix_name : str
        The name of a stored pulsar correlation matrix.

    Returns
    -------
    corr_matrix : array
        (npsrs x npsrs) covariance matrix corresponding
        to supplied name, where npsrs is the number of
        pulsars in the array.
    """
    if correlation_matrix_name == 'HD':
        corr_matrix = jnp.array(hdcorrmat(data.psrpos))
        return corr_matrix
    elif correlation_matrix_name == 'CURN':
        corr_matrix = jnp.eye(data.npsrs)
        return corr_matrix
    else:
        raise ValueError(
            f'Unknown pulsar correlation matrix "{correlation_matrix_name}".'
        )

