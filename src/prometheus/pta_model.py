'''Store full PTA modeling including posterior and sampling model.'''


from jax import vmap
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist
from typing import Optional, Callable

from .spectral_models import SpectralModel, IndependentSpectralModel, CommonSpectralModel
from .deterministic_models import DeterministicModel
from . import utilities as utils
from . import posterior



class PTAModel:

    """
    A general PTA model which accepts constituent models for stochastic
    and deterministic processes, and constructs the full joint posterior
    function and sampling model.

    The PTAModel has two operational modes: 'standard' and 'custom'.
    Both modes are compatible with deterministic models.
    
    In 'standard' mode, the user must supply an input for the 'psr_model'
    and 'gwb_model'. These act (usually) as the pulsar noise and GWB
    model, respectively.

    In 'custom' mode, the user supplies neither a 'psr_model' nor a 'gwb_model'.
    Instead, the user supplies a more general 'spectral_model' which
    represents all Gaussian processes modeled in the array. As the name
    suggests, 'custom' mode allows for much more customizable models, but
    can require more work to implement. See the advanced modeling example
    notebooks for examples using 'custom' mode.

    Required Attributes
    -------------------
    psr_model : IndependentSpectralModel
        An instance of an IndependentSpectralModel class found in spectral_models.py.
        This usually represents the pulsar noise model.
    gwb_model : CommonSpectralModel
        An instance of the CommonSpectralModel class found in spectral_models.py.
        This usually represents the GWB model.
    spectral_model : SpectralModel
        An instance of the SpectralModel class found in spectral_models.py. This
        is the more customizable model for advanced users which represents all
        Gaussian processes. See the advanced modeling example notebooks.
    det_model : DeterministicModel
        An instance of the DeterministicModel class found in deterministic_models.py.
        If None, a null deterministic model is used which has no parameters and induces
        no timing delays.
    add_ln_factor : Callable
        A function which takes all joint model parameters as input and outputs an
        additional (natural-log) factor to include in the posterior. By default,
        Prometheus uses uniform priors for all model parameters (except pulsar distance
        in deterministic models which uses a normal prior). So if a user wants to use
        alternative priors, they can provide that weighting here. See the advanced
        modeling example notebooks. By default, no extra factor is included in the posterior.
    """

    def __init__(self,
                 psr_model : Optional[IndependentSpectralModel] = None,
                 gwb_model : Optional[CommonSpectralModel] = None,
                 spectral_model : Optional[SpectralModel] = None,
                 det_model : Optional[DeterministicModel] = None):
        
        self.psr_model = psr_model
        self.gwb_model = gwb_model
        self.spectral_model = spectral_model
        self.det_model = det_model
        
        # determine which mode: 'standard' or 'custom'
        if self.spectral_model is not None:
            if self.psr_model is not None or self.gwb_model is not None:
                raise ValueError('Use either spectral_model OR psr_model+gwb_model, not both.')
            self.mode = 'custom'
            self.data = self.spectral_model.data
        else:
            if psr_model is None or gwb_model is None:
                raise ValueError('psr_model and gwb_model must both be provided.')
            self.mode = 'standard'
            self.data = self.psr_model.data

        # useful attributes
        self.psr_names = self.data.psr_names
        self.npsrs = self.data.npsrs
        self.ncomponents = self.data.ncomponents

        # check if user specified a deterministic model
        if self.det_model is None:
            self.include_det_model = False
        else:
            self.include_det_model = True


    def sampling_model(self, T=1.0):
        """
        Construct the NumPyro probabilistic sampling model.

        Note all parameters assume a uniform prior. If 'log'-parameters are
        supplied, they also get a uniform prior (so the argument of the log
        gets a log-uniform prior). The exception are pulsar distance parameters
        used in some deterministic models which use a normal prior.

        If the user desires different priors, they can supply those in the
        'additional_ln_factor' input to the PTAModel object. See the advanced
        modeling examples.
        """

        # standard mode: user supplies a pulsar noise model and separate GWB model
        if self.mode == 'standard':

            # pulsar noise hyper-parameters
            psr_params = numpyro.sample(name=self.psr_model.name,
                                    fn=dist.Uniform(low=self.psr_model.param_mins,  
                                                    high=self.psr_model.param_maxs,).expand([self.npsrs, self.psr_model.nparams_base]))

            # build pulsar noise prior covariance
            psr_phi_cube = self.psr_model.get_phi_cube(psr_params, self.data.freqs)

            # additional ln-factor for pulsar parameters specified by user
            if self.psr_model.additional_ln_factor is not None:
                ln_factor_psr = self.psr_model.additional_ln_factor(psr_params)
                numpyro.factor('additional_ln_factor_psr', ln_factor_psr)

            # GWB hyper-parameters
            gwb_params = numpyro.sample(name=self.gwb_model.name,
                                        fn=dist.Uniform(low=self.gwb_model.param_mins,
                                                        high=self.gwb_model.param_maxs))

            # build GWB prior covariance
            gwb_phi_cube = self.gwb_model.get_phi_cube(gwb_params, self.data.freqs)

            # additional ln-factor for GWB parameters specified by user
            if self.gwb_model.additional_ln_factor is not None:
                ln_factor_gwb = self.gwb_model.additional_ln_factor(gwb_params)
                numpyro.factor('additional_ln_factor_gwb', ln_factor_gwb)

            # combined prior covariance matrix
            phi_cube = psr_phi_cube + gwb_phi_cube
        
        
        # custom mode: user supplies single spectral model
        else:

            # sample hyper-parameters of custom spectral model
            spectral_params = numpyro.sample(name=self.spectral_model.name,
                                             fn=dist.Uniform(low=self.spectral_model.param_mins,
                                                             high=self.spectral_model.param_maxs))
    
            # build covariance matrix from hyper-parameters
            phi_cube = self.spectral_model.get_phi_cube(spectral_params, self.data.freqs)

            # additional ln-factor for spectral parameters specified by user
            if self.spectral_model.additional_ln_factor is not None:
                ln_factor_spectral = self.spectral_model.additional_ln_factor(spectral_params)
                numpyro.factor('additional_ln_factor_spectral', ln_factor_spectral)
        
        # sample parameters in deterministic signal model
        if self.include_det_model: # if a non-trivial deterministic model is provided
            
            # parameters of deterministic model use uniform priors
            det_params = numpyro.sample(self.det_model.name, dist.Uniform(self.det_model.param_mins,  
                                                                          self.det_model.param_maxs)) 
            if self.det_model.with_psr_params:  # if pulsar parameters are required
                psr_phases = numpyro.sample('psr_phases', dist.Uniform(0., 2. * jnp.pi).expand([self.npsrs]))
                standard_psr_dists = numpyro.sample('standard_psr_dists', dist.Normal().expand([self.npsrs]))
                psr_dists = numpyro.deterministic('psr_dists', self.data.psr_dists_measured + standard_psr_dists * self.data.psr_dists_std)
            else:   # if no pulsar parameters are needed for deterministic model
                psr_phases = None
                psr_dists = None
            
            # get Fourier coefficients of deterministic signal
            a_det = self.det_model.get_coeffs_func(det_params, psr_phases, psr_dists)

            # inner product needed for likelihood and standardizing transformation
            TNTDas = vmap(lambda x, y: jnp.dot(x, y))(self.data.TNTDs / T, a_det)

            # additional ln-factor for deterministic parameters specified by user
            if self.det_model.additional_ln_factor is not None:
                ln_factor_det = self.det_model.additional_ln_factor(det_params, psr_phases, psr_dists)
                numpyro.factor('additional_ln_factor_det', ln_factor_det)

        # sample in the space of "whitened" Fourier coefficients
        z = numpyro.sample('z', dist.Normal().expand([self.npsrs, self.ncomponents]))

        # NumPyro adds a standard normal probability density for the line above
        # this is not in the posterior, so we need to subtract it manually
        numpyro.factor('ln_inverse_normal_correction', 0.5 * jnp.sum(z**2))

        # get Cholesky, inverse, and log-determinant of prior covariance
        phi_chol_factors, phiinvs, phi_ln_dets = posterior.cholesky_inverse_det_phi(phi_cube)

        # do standardizing transform
        FNr_for_transform = self.data.FNrs - TNTDas if self.include_det_model else self.data.FNrs
        a, Sigma_inv_L = posterior.standardizing_transform(z=z,
                                                           phiinvs=phiinvs,
                                                           FNFs=self.data.FNFs,
                                                           FNrs=FNr_for_transform,
                                                           T=T)

        # ln-determinant Jacobian of standardizing transform
        lndet_Jac = -jnp.sum(jnp.log(jnp.diagonal(Sigma_inv_L, axis1=1, axis2=2)))
        numpyro.factor('ln_Jac', lndet_Jac)

        # save transformed coefficients (which obey spectral models)
        numpyro.deterministic('a', a)

        # evaluate the likelihood
        lnlike_val = posterior.ln_likelihood(a, self.data.FNFs, self.data.FNrs, T)
        numpyro.factor('ln_likelihood', lnlike_val)

        # evaluate the prior
        lnprior_val = posterior.ln_normal_prior(a, phiinvs, phi_ln_dets)
        numpyro.factor('ln_prior', lnprior_val)

        # extra terms in log-likelihood from deterministic signal
        if self.include_det_model:
            lnlike_det_add = posterior.ln_likelihood_det_addition(a_det, a, self.data.TDNrs,
                                                                  self.data.TNTDs, self.data.TDNTDs,
                                                                  TNTDas, T)
            numpyro.factor('ln_likelihood_det_addition', lnlike_det_add)


    def marginalized_sampling_model(self):
        """
        NumPyro sampling model with marginalized pulsar noise coefficients.
        """

        # pulsar noise hyper-parameters
        psr_params = numpyro.sample(name=self.psr_model.name,
                                fn=dist.Uniform(low=self.psr_model.param_mins,  
                                                high=self.psr_model.param_maxs,).expand([self.npsrs, self.psr_model.nparams_base]))

        # GWB hyper-parameters
        gwb_params = numpyro.sample(name=self.gwb_model.name,
                                    fn=dist.Uniform(low=self.gwb_model.param_mins,
                                                    high=self.gwb_model.param_maxs))
        
        # draw whitened Fourier coefficients for the background
        z = numpyro.sample(name='z', fn=dist.Normal().expand([self.npsrs, self.ncomponents]))

        # NumPyro adds a standard normal probability density for the line above
        # this is not in the posterior, so we need to subtract it manually
        numpyro.factor('ln_inverse_normal_correction', 0.5 * jnp.sum(z**2))
        
        # constants needed for likelihood evaluation
        FNrs = self.data.FNrs
        FNFs = self.data.FNFs

        # prior covariance for pulsar noise coefficients (Np, 2Nf, 2Nf)
        # TODO: replace with non-diagonal phi from window effects
        psr_phi_diags = vmap(self.psr_model.get_phi_diag_func, in_axes=(0, None))(psr_params, self.data.freqs)
        psr_phi = vmap(jnp.diag)(psr_phi_diags)
        psr_phi_chol_factors = vmap(lambda x: jsl.cho_factor(x, lower=True))(psr_phi)
        psr_phiinvs = vmap(lambda cf: jsl.cho_solve((cf[0], True),
                                                jnp.identity(cf[0].shape[0])))(psr_phi_chol_factors)
        psr_phi_ln_dets = vmap(lambda cf: 2 * jnp.sum(jnp.log(jnp.diag(cf[0]) / utils.renorm)))(psr_phi_chol_factors)

        # posterior covariance for pulsar noise coefficients (Np, 2Nf, 2Nf)
        psr_sigma_inv = FNFs + psr_phiinvs
        psr_sigma_inv_chol_factors = vmap(lambda x: jsl.cho_factor(x, lower=True))(psr_sigma_inv)
        psr_sigma_inv_ln_dets = vmap(lambda cf: 2 * jnp.sum(jnp.log(jnp.diag(cf[0]) / utils.renorm)))(psr_sigma_inv_chol_factors)

        # additional terms needed for likelihood and standardizing transform
        Linv_FNFs = vmap(lambda cf, FNF: jsl.solve_triangular(cf[0], FNF, lower=True))(psr_sigma_inv_chol_factors, FNFs)
        Linv_FNrs = vmap(lambda cf, FNr: jsl.solve_triangular(cf[0], FNr, lower=True))(psr_sigma_inv_chol_factors, FNrs)
        FNF_psr_sigma_FNFs = vmap(lambda A: A.T @ A)(Linv_FNFs)
        rNF_psr_sigma_FNrs = vmap(lambda v: v.T @ v)(Linv_FNrs)
        rNF_psr_sigma_FNFs = vmap(lambda v, A: v.T @ A)(Linv_FNrs, Linv_FNFs)

        # covariance of GWB Fourier coefficients
        # TODO: replace with non-diagonal phi from window effects
        gwb_phi_spec_diag = self.gwb_model.get_phi_diag_func(gwb_params, self.data.freqs)   # (2 * Nf)
        gwb_phi_spec = jnp.diag(gwb_phi_spec_diag)    # (2 * Nf, 2 * Nf)
        gwb_phi_psr_corr = self.gwb_model.correlation_matrix    # (Np, Np)
        # gwb_phi_inv_spec = jnp.linalg.inv(gwb_phi_spec)    # (2 * Nf, 2 * Nf)
        gwb_phi_inv_spec = jnp.diag(1. / gwb_phi_spec_diag)
        # gwb_phi_inv_psr_corr = jnp.linalg.inv(gwb_phi_psr_corr)    # (Np, Np)
        gwb_phi_psr_corr_cho_factors = jsl.cho_factor(gwb_phi_psr_corr, lower=True)
        gwb_phi_inv_psr_corr = jsl.cho_solve((gwb_phi_psr_corr_cho_factors[0], True),
                                             jnp.identity(gwb_phi_psr_corr_cho_factors[0].shape[0]))
        gwb_phi_inv = jnp.kron(gwb_phi_inv_psr_corr, gwb_phi_inv_spec)    # (Np * 2 * Nf, Np * 2 * Nf)
        
        # log-determinant of GWB covariance
        gwb_phi_spec_lndet = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(gwb_phi_spec)) / utils.renorm))
        gwb_phi_psr_corr_lndet = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(gwb_phi_psr_corr)) / utils.renorm))
        gwb_phi_lndet = self.npsrs * gwb_phi_spec_lndet + self.ncomponents * gwb_phi_psr_corr_lndet

        # do standardizing transform
        gwb_sigma_inv = FNFs + gwb_phi_inv_spec[None, :, :] - FNF_psr_sigma_FNFs
        Sigma_inv_L = vmap(lambda x: jsl.cholesky(x, lower=True))(gwb_sigma_inv)
        y = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=0))(Sigma_inv_L, -FNrs + rNF_psr_sigma_FNFs)
        a_hat = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Sigma_inv_L, y)
        Lz = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Sigma_inv_L, z)
        a_gwb = a_hat + Lz
        numpyro.deterministic('a_gwb', a_gwb)

        # ln-determinant Jacobian of standardizing transform
        lndet_Jac = -jnp.sum(jnp.log(jnp.diagonal(Sigma_inv_L, axis1=1, axis2=2)))
        numpyro.factor('ln_Jac', lndet_Jac)

        # evaluate the likelihood
        Ws = FNFs - FNF_psr_sigma_FNFs
        Vs = -FNrs + rNF_psr_sigma_FNFs
        ln_likelihood_val = -0.5 * jnp.sum(vmap(lambda a, W: jnp.dot(a, jnp.dot(W, a)))(a_gwb, Ws))
        ln_likelihood_val += jnp.sum(vmap(lambda a, V: jnp.dot(a, V))(a_gwb, Vs))
        ln_likelihood_val += 0.5 * jnp.sum(rNF_psr_sigma_FNrs)
        ln_likelihood_val += -0.5 * jnp.sum(psr_phi_ln_dets) - 0.5 * jnp.sum(psr_sigma_inv_ln_dets)
        numpyro.factor('lnlike', ln_likelihood_val)

        # evaluate the prior
        ln_prior_val = -0.5 * jnp.dot(a_gwb.flatten(), jnp.dot(gwb_phi_inv, a_gwb.flatten()))
        ln_prior_val += -0.5 * gwb_phi_lndet
        numpyro.factor('lnprior', ln_prior_val)    


