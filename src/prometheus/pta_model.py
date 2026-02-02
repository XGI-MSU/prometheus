'''Store full PTA modeling including posterior and sampling model.'''


from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist
from typing import Optional, Callable

from .spectral_models import SpectralModel, IndependentSpectralModel, CommonSpectralModel
from .deterministic_models import DeterministicModel, build_null_deterministic_model
from . import utilities as utils



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
                 det_model : Optional[DeterministicModel] = None,
                 add_ln_factor : Optional[Callable] = None):
        
        self.psr_model = psr_model
        self.gwb_model = gwb_model
        self.spectral_model = spectral_model
        self.det_model = det_model
        self.add_ln_factor = add_ln_factor
        
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

        # if no deterministic model is supplied, use null model
        if self.det_model is None:
            det_model = build_null_deterministic_model(self.data)
            self.det_model = det_model

        # posterior to test/time
        # technically this can be sampled, but the sampling model below
        # codes this up for us in NumPyro's probabilistic programming language
        self.ln_posterior = self.build_posterior()
    

    def ln_posterior_components(self, xi, phi_cube, det_params, psr_phases, psr_dists):
        """
        Evaluates the components of the full joint posterior. The sampled posterior is equivalent
        to that of Enterprise and other PTA analysis softwares. The implementation differences are:
            - the Fourier coefficients are sampled numerically, rather than analytically marginalized
            - the Fourier coefficients are sampled under a standardizing transform
            - the deterministic signal is represented in a Fourier basis.
        Note these 'differences' are purely in implementation. After sampling, we recover
        a posterior distribution identical to that of other fully joint analyses.
        
        Parameters
        ----------
        self : PTAModel
            Instance of the PTAModel class.
        xi : array
            Array of shape (Npsrs, 2*Nf) where Npsrs are the number of pulsars in the array
            and Nf are the number of frequency bins modeled. These are the 'standardized'
            Fourier coefficients, drawn from a standard normal distribution. Below they
            are transformed with the standardizing transform to obey (approximately) the
            spectrum given by our spectral models.
        phi_cube : array
            Array of shape (2*Nf, Npsrs, Npsrs) where Nf is the number of frequency bins
            modeled and Npsrs are the number of pulsars in the array. This is the prior
            covariance matrix for the Fourier coefficients. The (i, j, k) element of this
            array is the covariance of the ith Fourier coefficient between pulsars j and k.
            This array depends on a set of spectral-hyper-paramters and is constructed in
            the 'sampling_model' method below.
        det_params : array
            Array of deterministic parameters. If the 'null' deterministic model is used,
            the parameters are NoneType.
        psr_phases : array
            Array of shape (npsrs,) where npsrs is the number of puslars in the array.
            The array is the phase of the deterministic GW at each pulsar. If the
            deterministic 'null' model or a DeterministicModel with `with_psr_params=False'
            is used, then these parameters are NoneType.
        psr_dists : array
            Array of shape (npsrs,) where npsrs is the number of puslars in the array.
            The array is the distance to each pulsar. If the deterministic 'null' model
            or a DeterministicModel with `with_psr_params=False' is used, then 
            these parameters are NoneType.
        
        Returns
        -------
        (logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a_stochastic) : tuple
            The components of the full joint posterior, except the last element which is
            the transformed Fourier coefficients (transformed to obey approximately the spectral models).
            The sum of these components is the value of the posterior density evaluated at the input
            parameters.
        """

        # Use vmap to vectorize cho_factor and cho_solve across the batch (phi_cube)
        phi_chol_factors = vmap(lambda x: jsl.cho_factor(x, lower=True))(phi_cube)
        phiinvs = vmap(lambda cf: jsl.cho_solve((cf[0], True), jnp.identity(cf[0].shape[0])))(phi_chol_factors)
        philogdets = 2*jnp.sum(jnp.log(jnp.diagonal(phi_chol_factors[0], axis1=1, axis2=2)/utils.renorm), axis=1)

        # Get the CURN-based Cholesky transform quantities
        phiinv_vecs_c_fp = jnp.diagonal(phiinvs, axis1=1, axis2=2)     # nfreqs x npsrs
        phiinv_c_pf = jnp.zeros((self.data.phiinv_0_vecs_j.shape[0], self.data.phiinv_0_vecs_j.shape[1], self.data.phiinv_0_vecs_j.shape[1]))
        phiinv_c_pf = phiinv_c_pf.at[:, self.data.ii_diag_pf, self.data.ii_diag_pf].set(phiinv_vecs_c_fp.T)
        Sigma_inv_c = self.data.Sigma_0_inv_jc + phiinv_c_pf - self.data.phiinv_0_cube_pf

        # get frequency-domain representation of deterministic signal (npsrs, Na_det)
        a_det = self.det_model.get_coeffs_func(det_params, psr_phases, psr_dists)  

        # Do decentering transformation with deterministic correction
        Lc = vmap(lambda x: jsl.cholesky(x, lower=True))(Sigma_inv_c)
        TNTDas = vmap(lambda x, y: jnp.dot(x, y))(self.data.TNTDs, a_det)
        Lca = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=0))(Lc, self.data.Si0_a_hat_j_pf - TNTDas)    # Lc @ TNr
        LLca = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Lc, Lca)  # a-hat
        am = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Lc, xi) # L @ xi, utils.renormalized already
        a_stochastic = am + LLca

        # The Jacobian
        logJac = -jnp.sum(jnp.log(jnp.diagonal(Lc, axis1=1, axis2=2)))

        # The value of the actual likelihood
        loglik_aSa = -0.5*jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stochastic-self.data.a_hat_2d_pf,
                                                                                self.data.Sigma_0_inv_jc))
        loglik_ld = -0.5*jnp.sum(self.data.Sigma_0_logdet_j)

        # In NumPyro we define xi ~ N(0, 1). But that's just a crutch. Actually it's a
        # MvNormal. So, sneakily we just remove the logP of xi from NumPyro here
        loglik_chol = 0.5 * jnp.sum(xi * xi)

        # The prior
        logprior_aPa = -0.5*jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_stochastic.T, phiinvs - self.data.phiinv_0_cube_fp))
        logprior_ld = -0.5*jnp.sum(philogdets) + 0.5*self.data.phiinv_logdet_0_j

        # deterministic contribution to likelihood
        loglik_aSa += jnp.sum(vmap(lambda x, y: jnp.dot(x, y))(a_det, self.data.TDNrs))
        loglik_aSa += -jnp.sum(vmap(lambda x, y: jnp.dot(x, y))(a_stochastic, TNTDas))
        loglik_aSa += -0.5 * jnp.sum(vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_det, self.data.TDNTDs))

        return logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a_stochastic


    def addition(self, logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld):
        """
        Add up the components of the posterior obtained from 'ln_posterior_components' to
        evaluate the full joint posterior.
        """
        return logJac + loglik_aSa + loglik_ld + loglik_chol + logprior_aPa + logprior_ld

    def sampling_model(self):
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

        # sample standardized Fourier coefficients
        xi = numpyro.sample('xi', dist.Normal().expand([self.npsrs, self.ncomponents]))

        # deterministic model
        if not self.det_model.null: # if a non-trivial deterministic model is provided
            # sample parameters of deterministic signal
            det_params = numpyro.sample(self.det_model.name, dist.Uniform(self.det_model.param_mins,  
                                                                          self.det_model.param_maxs)) 
            if self.det_model.with_psr_params:  # if pulsar parameters are required
                psr_phases = numpyro.sample('psr_phases', dist.Uniform(0., 2. * jnp.pi).expand([self.npsrs]))
                standard_psr_dists = numpyro.sample('standard_psr_dists', dist.Normal().expand([self.npsrs]))
                psr_dists = numpyro.deterministic('psr_dists', self.data.psr_dists_measured + standard_psr_dists * self.data.psr_dists_std)
            else:   # if no pulsar parameters are needed for deterministic model
                psr_phases = None
                psr_dists = None
        else:   # if no deterministic model is supplied
            det_params = None
            psr_phases = None
            psr_dists = None

        # standard mode requires a pulsar noise and GWB model
        if self.mode == 'standard':
            # pulsar noise hyper-parameters
            pn_params = numpyro.sample(name=self.psr_model.name,
                                    fn=dist.Uniform(low=self.psr_model.param_mins,  
                                                    high=self.psr_model.param_maxs,).expand([self.npsrs, self.psr_model.nparams_base]))
            
            # GWB hyper-parameters
            gwb_params = numpyro.sample(name=self.gwb_model.name,
                                        fn=dist.Uniform(low=self.gwb_model.param_mins,
                                                        high=self.gwb_model.param_maxs))

            # build covariance matrix from hyper-parameters
            phi_cube = (self.psr_model.get_phi_cube(pn_params, self.data.freqs)
                        + self.gwb_model.get_phi_cube(gwb_params, self.data.freqs))
                
        else:   # 'custom' mode enabled
            # sample hyper-parameters of custom spectral model
            spectral_params = numpyro.sample(name=self.spectral_model.name,
                                             fn=dist.Uniform(low=self.spectral_model.param_mins,
                                                             high=self.spectral_model.param_maxs))
    
            # build covariance matrix from hyper-parameters
            phi_cube = self.spectral_model.get_phi_cube(spectral_params, self.data.freqs)
        
        # get components of the likelihood
        logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                phi_cube,
                                                                                                                det_params,
                                                                                                                psr_phases,
                                                                                                                psr_dists)
        
        # save transformed coefficients (which obey spectral models)
        numpyro.deterministic('a', a)

        # evaluate the full joint posterior
        ln_posterior_value = self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
        numpyro.factor('ln_posterior', ln_posterior_value)

        # add additional ln-factor if provided by user
        # (this is a mess, but it's supposed to make it easy for a user to supply
        # additional posterior corrections using **only** their specified parameters)
        # I don't want to use dictionary unwrapping... I **think** this is faster.
        if self.add_ln_factor is not None:
            if self.mode == 'standard':
                if self.det_model.null:
                    numpyro.factor('additional_ln_factor', self.add_ln_factor(pn_params, gwb_params))
                else:
                    if self.det_model.with_psr_params:
                        numpyro.factor('additional_ln_factor', self.add_ln_factor(pn_params, gwb_params, det_params, psr_phases, psr_dists))
                    else:
                        numpyro.factor('additional_ln_factor', self.add_ln_factor(pn_params, gwb_params, det_params))
            if self.mode == 'custom':
                if self.det_model.null:
                    numpyro.factor('additional_ln_factor', self.add_ln_factor(spectral_params))
                else:
                    if self.det_model.with_psr_params:
                        numpyro.factor('additional_ln_factor', self.add_ln_factor(spectral_params, det_params, psr_phases, psr_dists))
                    else:
                        numpyro.factor('additional_ln_factor', self.add_ln_factor(spectral_params, det_params))

        return self.sampling_model
    

    def get_param_names_and_shapes(self):
        """
        Get a dictionary whose keys are the names of parameters
        used in the PTAModel and values are the required shapes
        of the parameters.
        """
        param_dict = dict()
        param_dict['xi'] = (self.npsrs, self.ncomponents)
        if self.mode == 'standard':
            param_dict[self.psr_model.name] = (self.npsrs, self.psr_model.nparams_base)
            param_dict[self.gwb_model.name] = (self.gwb_model.nparams_base,)
        else:
            param_dict[self.spectral_model.name] = (self.psr_model.nparams_base,)
        if not self.det_model.null:
            param_dict[self.det_model.name] = (self.det_model.nparams,)
            if self.det_model.with_psr_params:
                param_dict['psr_phases'] = (self.npsrs,)
                param_dict['psr_dists'] = (self.npsrs,)
        return param_dict
    

    def build_posterior(self):
        """
        Method to build the posterior, so the user only has to implement the
        parameters **they know about**. For example, if no deterministic model
        is in use, the user doesn't have to make up parameter values to feed 
        the 'null' deterministic model.

        This posterior can technically be sampled, but a NumPyro compatible
        sampling model is built above in the 'sampling_model' method.

        This formatting is a bit of a mess. To keep the posterior efficient,
        we want to avoid parameter dictionary unwrapping. This unfortunately
        requires us to branch to all different model types.
        -- if someone knows a better way to do this, please let me know...
        """
        if self.mode == 'standard':
            def get_phi_cube_for_posterior(pn_params, gwb_params):
                return (self.psr_model.get_phi_cube(pn_params, self.data.freqs)
                        + self.gwb_model.get_phi_cube(gwb_params, self.data.freqs))
            if self.det_model.null:
                def ln_posterior(xi, pn_params, gwb_params):
                    phi_cube = get_phi_cube_for_posterior(pn_params, gwb_params)
                    logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                phi_cube,
                                                                                                                None,
                                                                                                                None,
                                                                                                                None)
                    return self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
            else:
                if not self.det_model.with_psr_params:
                    def ln_posterior(xi, pn_params, gwb_params, det_params):
                        phi_cube = get_phi_cube_for_posterior(pn_params, gwb_params)
                        logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                    phi_cube,
                                                                                                                    det_params,
                                                                                                                    None,
                                                                                                                    None)
                        return self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
                else:
                    def ln_posterior(xi, pn_params, gwb_params, det_params, psr_phases, psr_dists):
                        phi_cube = get_phi_cube_for_posterior(pn_params, gwb_params)
                        logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                    phi_cube,
                                                                                                                    det_params,
                                                                                                                    psr_phases,
                                                                                                                    psr_dists)
                        return self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
        else:
            def get_phi_cube_for_posterior(spectral_params):
                return self.spectral_model.get_phi_cube(spectral_params, self.data.freqs)
            if self.det_model.null:
                def ln_posterior(xi, spectral_params):
                    phi_cube = get_phi_cube_for_posterior(spectral_params)
                    logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                phi_cube,
                                                                                                                None,
                                                                                                                None,
                                                                                                                None)
                    return self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
                return ln_posterior
            else:
                if not self.det_model.with_psr_params:
                    def ln_posterior(xi, spectral_params, det_params):
                        phi_cube = get_phi_cube_for_posterior(spectral_params)
                        logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                    phi_cube,
                                                                                                                    det_params,
                                                                                                                    None,
                                                                                                                    None)
                        return self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
                else:
                    def ln_posterior(xi, spectral_params, det_params, psr_phases, psr_dists):
                        phi_cube = get_phi_cube_for_posterior(spectral_params)
                        logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld, a = self.ln_posterior_components(xi,
                                                                                                                    phi_cube,
                                                                                                                    det_params,
                                                                                                                    psr_phases,
                                                                                                                    psr_dists)
                        return self.addition(logJac, loglik_aSa, loglik_ld, loglik_chol, logprior_aPa, logprior_ld)
        return jit(ln_posterior)


    def draw_params_from_prior(self, seed=0):
        """
        Draw a set of parameters from their prior. This is not
        efficient and is only used to test the posterior evaluation.
        In practice, the NumPyo sampling model does this for us.

        All parameters use a uniform prior. Except:
            - xi: use a standard normal prior
            - psr_dists: use a normal prior
        """
        prng_key = jr.key(seed)
        param_dict = dict()
        
        # add standardized Fourier coefficients
        key = jr.split(prng_key)[1]
        xi = jr.normal(key=key,
                       shape=(self.npsrs, self.ncomponents))
        param_dict['xi'] = xi

        # spectral parameters
        if self.mode == 'standard':
            # pulsar noise parameters
            key = jr.split(key)[1]
            pn_params = jr.uniform(key=key,
                                   shape=(self.npsrs, self.psr_model.nparams_base),
                                   minval=self.psr_model.param_mins,
                                   maxval=self.psr_model.param_maxs)
            param_dict[self.psr_model.name] = pn_params
            # GWB parameters
            key = jr.split(key)[1]
            gwb_params = jr.uniform(key=key,
                                    shape=(self.gwb_model.nparams_base,),
                                    minval=self.gwb_model.param_mins,
                                    maxval=self.gwb_model.param_maxs)
            param_dict[self.gwb_model.name] = gwb_params
        else:
            key = jr.split(key)[1]
            spectral_params = jr.uniform(key=key,
                                         shape=(self.spectral_model.nparams_base,),
                                         minval=self.spectral_model.param_mins,
                                         maxval=self.spectral_model.param_maxs)
            param_dict[self.spectral_model.name] = spectral_params
        
        # deterministic parameters
        if not self.det_model.null:
            key = jr.split(key)[1]
            det_params = jr.uniform(key=key,
                                    shape=(self.det_model.nparams,),
                                    minval=self.det_model.param_mins,
                                    maxval=self.det_model.param_maxs)
            param_dict[self.det_model.name] = det_params

            # pulsar parameters
            if self.det_model.with_psr_params:
                # pulsar phases
                key = jr.split(key)[1]
                psr_phases = jr.uniform(key=key,
                                        shape=(self.npsrs,),
                                        minval=jnp.zeros((self.npsrs,)),
                                        maxval=jnp.ones((self.npsrs,)) * 2 * jnp.pi)
                param_dict['psr_phases'] = psr_phases
                # pulsar distances
                key = jr.split(key)[1]
                psr_dists_standard = jr.normal(key=key,
                                               shape=(self.npsrs,))
                psr_dists = (self.data.psr_dists_measured
                             + self.data.psr_dists_std * psr_dists_standard)
                param_dict['psr_dists'] = psr_dists
        return param_dict

