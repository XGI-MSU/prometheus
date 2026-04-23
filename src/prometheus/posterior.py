'''Utility functions for likelihood, prior calculations.'''

from jax import vmap
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.scipy.stats import norm
from . import utilities as utils



def cholesky_inverse_det_phi(phi_cube):
    """
    Compute the Cholesky decomposition, inverse, and ln-determinant
    of a covariance matrix.
    
    Parameters
    ----------
    phi_cube : array
        Array of shape (2Nf, Np, Np) where Nf is the number of frequency bins
        and Np is the number of pulsars. The (2i, j, k) element is the covariance
        between pulsar j and pulsar k at frequency bin 2i.

    Returns
    -------
    phi_chol_factors, phiinvs, phi_ln_dets : tuple
        Tuple containing the Cholesky factors, inverses, and log-determinants
        of the covariance matrices, batched over frequency bins.

    """
    phi_chol_factors = vmap(lambda x: jsl.cho_factor(x, lower=True))(phi_cube)
    phiinvs = vmap(lambda cf: jsl.cho_solve((cf[0], True),
                                            jnp.identity(cf[0].shape[0])))(phi_chol_factors)
    phi_ln_dets = 2 * jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0],
                                                   axis1=1, axis2=2) / utils.renorm), axis=1)
    return phi_chol_factors, phiinvs, phi_ln_dets


def ln_normal_prior(a, phiinvs, phi_ln_dets):
    """
    Evaluate the zero-mean multivariate normal prior for
    the Fourier coefficients.
    
    Parameters
    ----------
    a : array
        Array of shape (Np, 2Nf) where Nf is the number of frequency bins
        and Np is the number of pulsars. The (i, 2j) element is the Fourier
        coefficient of the ith pulsar at frequency bin j. The Fourier basis
        is ordered sine, cosine per frequency bin.
    phiinvs : array
        Array of shape (2Nf, Np, Np). The inverse of the prior covariance matrix, phi.
    phi_ln_dets : array
        Array of shape (2Nf,) containing the ln-determinants of the prior covariance matrix,
        batched over frequency bins.

    Returns
    -------
    lnprior_value : float
        The ln-prior value of the Fourier coefficients.
    """
    a_phiinv_a = jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a.T, phiinvs))
    ln_det_phi = jnp.sum(phi_ln_dets)
    lnprior_value = -0.5 * (a_phiinv_a + ln_det_phi)
    return lnprior_value


def ln_likelihood(a, FNFs, FNrs, T=1.0):
    """
    Evaluate the likelihood of the data, conditioned on the
    set of Fourier coefficients representing stochastic processes.
    
    Parameters
    ----------
    a : array
        Array of shape (Np, 2Nf) where Nf is the number of frequency bins
        and Np is the number of pulsars. The (i, 2j) element is the Fourier
        coefficient of the ith pulsar at frequency bin j. The Fourier basis
        is ordered sine, cosine per frequency bin.
    FNFs : array
        Array of shape (Np, 2Nf, 2Nf). Noise weighted inner product of Fourier design matrix
        with respect to itself.
    FNrs : array
        Array of shape (Np, 2Nf). Noise weighted inner product of Fourier design matrix
        with the timing residuals.
    
    T : float, optional
        Temperature of chain. Default is 1.0.

    Returns
    -------
    lnlike_value : float
        The ln-likelihood value at temperature T.
    """
    aFNr = jnp.sum(vmap(lambda x, y: jnp.dot(x, y))(a, FNrs))
    aFNFa = jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a, FNFs))
    lnlike_value = (aFNr - 0.5 * aFNFa) / T
    return lnlike_value


def ln_likelihood_det_addition(a_det, a_stochastic, TDNrs, TNTDs, TDNTDs, TNTDas, T=1.0):
    lnlike_det_add = jnp.sum(vmap(lambda x, y: jnp.dot(x, y))(a_det, TDNrs)) \
        + -jnp.sum(vmap(lambda x, y: jnp.dot(x, y))(a_stochastic, TNTDas)) \
        + -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_det, TDNTDs))
    return lnlike_det_add / T


def standardizing_transform(z, phiinvs, FNFs, FNrs, T=1.0):
    
    # get CURN Fourier coefficient (posterior) covariance and Cholesky
    phiinv_vecs_curn_fp = jnp.diagonal(phiinvs, axis1=1, axis2=2) # (2 * nfreqs, npsrs)
    Na, Np = phiinv_vecs_curn_fp.shape
    phiinv_curn_pf = jnp.zeros((Np, Na, Na))
    freq_ndxs = jnp.arange(Na)
    phiinv_curn_pf = phiinv_curn_pf.at[:, freq_ndxs, freq_ndxs].set(phiinv_vecs_curn_fp.T)
    Sigma_inv = FNFs / T + phiinv_curn_pf
    Sigma_inv_L = vmap(lambda x: jsl.cholesky(x, lower=True))(Sigma_inv)

    # get MAP coefficients
    y = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=0))(Sigma_inv_L, FNrs / T)
    a_hat = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Sigma_inv_L, y)

    # do standardizing transform
    Lz = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Sigma_inv_L, z)
    a = a_hat + Lz
    return a, Sigma_inv_L


def ln_p_PX(value, dist, err):
    """
    Parallax-based prior on pulsar distance (Arzoumanian+ 2023 Eq. 20)
    p(d) ∝ N(1/d | 1/dist, err/dist^2) * 1/d^2
    """
    pi = 1.0 / dist
    pi_err = err / dist**2
    inv_value = 1.0 / value
    z = (inv_value - pi) / pi_err
    lnprob = -0.5 * z**2 - jnp.log(jnp.sqrt(2 * jnp.pi) * pi_err * value**2)
    lnprob = jnp.where(value > 0, lnprob, -jnp.inf)
    return lnprob


def ln_p_DM(value, dist, err):
    """
    DM-based prior on pulsar distance (Arzoumanian+ 2023 Eq. 21)
    Flat between dist±err, Gaussian tails outside that range.
    """
    sigma = 0.25 * err
    boxheight = 1.0 / (2.0 * err)
    gaussheight = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    scale = boxheight / gaussheight
    area = 1.0 + scale  # normalization factor

    left  = norm.pdf(value, loc=dist - err, scale=sigma) * scale
    mid   = boxheight
    right = norm.pdf(value, loc=dist + err, scale=sigma) * scale

    y = jnp.where(
        value <= (dist - err), left,
        jnp.where(value < (dist + err), mid, right)
    )
    lnprob = jnp.log(y / area + 1e-12)
    lnprob = jnp.where(value > 0, lnprob, -jnp.inf)
    return lnprob
