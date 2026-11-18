'''Store full PTA modeling including posterior and sampling model.'''


import numpy as np
import jax
from jax import vmap
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from typing import Optional, Callable

from .spectral_models import SpectralModel, IndependentSpectralModel, CommonSpectralModel, DMSpectralModel
from .deterministic_models import DeterministicModel
from . import utilities as utils
from . import posterior


def _sample(name, fn):
    """
    numpyro.sample, rounded to float32: NumPyro keeps the position of the sampler
    in float64 (see __init__.py), but the posterior is evaluated in float32.
    """
    return numpyro.sample(name, fn).astype(jnp.float32)


@jax.custom_vjp
def batched_cholesky_identity_plus_gram(Bs):
    """
    Lower Cholesky factors L of M = I + B B^T for a batch Bs of shape (Np, k, n),
    returned in the dtype of Bs (float32).

    Loud signals (e.g. DM variations) give M a dynamic range of 1e10 and more. In
    float32, B B^T then loses the identity to rounding, and a Cholesky factorization of
    M is inaccurate or fails: the ln-likelihood picks up O(1) noise and NUTS step sizes
    collapse. The forward pass therefore forms M and factors it in float64, the only
    float64 computation of the package (see __init__.py), and rounds L to float32.

    The derivative of the Cholesky factorization is hand-written and evaluated in
    float32 from the rounded L (Murray 2016, arXiv:1602.07527): float64 matrix
    algebra is slow on consumer GPUs, and the derivative is ordered so that it forms
    neither M nor M_bar, so it does not inherit their dynamic range. Since the
    remaining error is a deterministic function of the position, NUTS stays exact.

    Returns
    -------
    Ls : array, shape (Np, k, k)
        Lower-triangular, M = L L^T.
    """
    B = Bs.astype(jnp.float64)
    with jax.default_matmul_precision('highest'):
        Ms = jnp.eye(B.shape[1], dtype=jnp.float64) + jnp.einsum('pkn,pjn->pkj', B, B)
    return vmap(jnp.linalg.cholesky)(Ms).astype(Bs.dtype)


def _batched_cholesky_identity_plus_gram_fwd(Bs):
    Ls = batched_cholesky_identity_plus_gram(Bs)
    return Ls, (Ls, Bs)


def _batched_cholesky_identity_plus_gram_bwd(res, Ls_bar):
    # for M = L L^T: M_bar = sym(L^{-T} Phi(L^T L_bar) L^{-1}), Phi = lower triangle
    # with halved diagonal; and for M = I + B B^T: B_bar = 2 M_bar B, evaluated as
    # B_bar = L^{-T} (P + P^T) L^{-1} B with P = Phi(L^T L_bar), solving for L^{-1} B
    # first (M_bar spans the dynamic range of M, while L^{-1} B has norm <= 1)
    Ls, Bs = res
    k = Ls.shape[1]
    with jax.default_matmul_precision('highest'):
        P = jnp.einsum('pji,pjk->pik', Ls, Ls_bar)
        P = jnp.tril(P) - 0.5 * P * jnp.eye(k, dtype=Ls.dtype)
        X = vmap(lambda L, B: jsl.solve_triangular(L, B, lower=True))(Ls, Bs)
        S = jnp.einsum('pkj,pjn->pkn', P + jnp.swapaxes(P, 1, 2), X)
        return (vmap(lambda L, S: jsl.solve_triangular(L, S, lower=True, trans='T'))(Ls, S),)


batched_cholesky_identity_plus_gram.defvjp(_batched_cholesky_identity_plus_gram_fwd,
                                           _batched_cholesky_identity_plus_gram_bwd)



class PTAModel:

    """
    A general PTA model which accepts constituent models for stochastic
    and deterministic processes, and constructs the full joint posterior
    function and sampling model.

    The PTAModel has two operational modes: 'standard' and 'custom'.
    Both modes are compatible with deterministic models.
    
    In 'standard' mode, the user supplies one or more per-pulsar processes
    ('psr_model', e.g. intrinsic red noise and DM variations) and optionally a
    common process ('gwb_model', e.g. the GWB). Each process lives on a named
    Fourier basis of the Data object (see data.Data and the 'basis' argument
    of the spectral models).

    In 'custom' mode, the user supplies neither a 'psr_model' nor a 'gwb_model'.
    Instead, the user supplies a more general 'spectral_model' which
    represents all Gaussian processes modeled in the array. As the name
    suggests, 'custom' mode allows for much more customizable models, but
    can require more work to implement. See the advanced modeling example
    notebooks for examples using 'custom' mode.

    Sampling models
    ---------------
    'shared' (sampling_model_shared): the common and per-pulsar processes are
        described by ONE set of Fourier coefficients on a shared basis, with prior
        covariance phi_psr + phi_gwb, all of which are sampled (through a
        standardizing transform). Requires every process to live on the 'red'
        basis without inter-frequency correlations. Always used in 'custom' mode.
    'marginalized' (sampling_model_marginalized): the per-pulsar and common
        coefficients are separate. The per-pulsar coefficients (of every basis,
        e.g. [F | F_dm]) are marginalized analytically, and only the common
        coefficients are sampled, under the standardizing transform conditional on
        the marginalized per-pulsar processes. Supports any number of per-pulsar
        processes on any bases (including bases on each pulsar's own span,
        Tspan='psr'), non-diagonal prior covariances (inter-frequency
        correlations), and a common process on its own basis.

    By default the 'shared' model is used whenever it applies and a common process
    is modeled, and the 'marginalized' model otherwise.

    Required Attributes
    -------------------
    psr_model : IndependentSpectralModel or list of IndependentSpectralModel
        The per-pulsar processes, e.g. the intrinsic pulsar noise (on the 'red'
        basis) and DM variations (on the 'dm' basis). Models on the same basis
        add their prior covariances.
    gwb_model : CommonSpectralModel
        An instance of the CommonSpectralModel class found in spectral_models.py.
        This usually represents the GWB model. If None (standard mode only), no
        common process is modeled.
    spectral_model : SpectralModel
        An instance of the SpectralModel class found in spectral_models.py. This
        is the more customizable model for advanced users which represents all
        Gaussian processes. See the advanced modeling example notebooks.
    det_model : DeterministicModel
        An instance of the DeterministicModel class found in deterministic_models.py.
        If None, a null deterministic model is used which has no parameters and induces
        no timing delays.
    dm_model : DMSpectralModel
        Convenience argument for a dispersion measure Gaussian process (DMGP): it is
        appended to the per-pulsar processes. If None, no DMGP is modeled (unless
        one is among the 'psr_model' list).
    sampling : str
        'shared', 'marginalized', or None (default) to choose automatically (see above).
    """

    def __init__(self,
                 psr_model : Optional[IndependentSpectralModel | list[IndependentSpectralModel]] = None,
                 gwb_model : Optional[CommonSpectralModel] = None,
                 spectral_model : Optional[SpectralModel] = None,
                 det_model : Optional[DeterministicModel] = None,
                 dm_model : Optional[DMSpectralModel] = None,
                 sampling : Optional[str] = None):

        # per-pulsar processes
        if psr_model is None:
            psr_models = []
        elif isinstance(psr_model, (list, tuple)):
            psr_models = list(psr_model)
        else:
            psr_models = [psr_model]
        if dm_model is not None:
            psr_models.append(dm_model)
        self.psr_models = psr_models
        self.psr_model = psr_models[0] if psr_models else None
        self.dm_model = dm_model if dm_model is not None else \
            next((m for m in psr_models if m.basis == 'dm'), None)
        self.gwb_model = gwb_model
        self.spectral_model = spectral_model
        self.det_model = det_model
        
        # determine which mode: 'standard' or 'custom'
        if self.spectral_model is not None:
            if self.psr_models or self.gwb_model is not None:
                raise ValueError('Use either spectral_model OR psr_model (+ gwb_model), not both.')
            self.mode = 'custom'
            self.data = self.spectral_model.data
        else:
            if not self.psr_models:
                raise ValueError('psr_model must be provided (gwb_model is optional).')
            self.mode = 'standard'
            self.data = self.psr_model.data

        # every constituent model must refer to the same Data object
        models = self.psr_models + [m for m in (self.gwb_model, self.spectral_model) if m is not None]
        if any(m.data is not self.data for m in models):
            raise ValueError('All spectral models must be built with the same Data object.')
        names = [m.name for m in models] + ([self.det_model.name] if self.det_model is not None else [])
        if len(set(names)) != len(names):
            raise ValueError(f'Model names must be unique, got {names}.')

        # check if user specified a common process (GWB) model
        self.include_gwb_model = self.gwb_model is not None
        self.include_dm_model = self.dm_model is not None

        # whether any prior covariance has inter-frequency correlations
        self.freq_corrs = any(m.freq_corrs for m in models) if self.mode == 'standard' else False

        # useful attributes
        self.psr_names = self.data.psr_names
        self.npsrs = self.data.npsrs
        self.ncomponents = self.data.ncomponents

        # check if user specified a deterministic model
        if self.det_model is None:
            self.include_det_model = False
        else:
            self.include_det_model = True

        # choose the sampling model
        shared_ok, reason = self._shared_compatible()
        if sampling is None:
            sampling = 'shared' if shared_ok and (self.mode == 'custom' or self.include_gwb_model) else 'marginalized'
        if sampling == 'shared':
            if not shared_ok:
                raise ValueError(f"The 'shared' sampling model does not apply: {reason}.")
            self.sampling_model = self.sampling_model_shared
        elif sampling == 'marginalized':
            if self.mode == 'custom':
                raise ValueError("Custom spectral models only support the 'shared' sampling model.")
            self.sampling_model = self.sampling_model_marginalized
        else:
            raise ValueError(f"sampling must be 'shared', 'marginalized' or None, got {sampling!r}.")
        self.sampling = sampling


    def _shared_compatible(self):
        """Whether the 'shared' sampling model applies, and if not, why."""
        if self.mode == 'custom':
            return True, ''
        if self.freq_corrs:
            return False, 'it does not support inter-frequency correlations'
        on_other_bases = [m.name for m in self.psr_models + [self.gwb_model]
                          if m is not None and m.basis != 'red']
        if on_other_bases:
            return False, f"processes {on_other_bases} do not live on the shared 'red' basis"
        return True, ''


    def sampling_model_shared(self, T=1.0):
        """
        NumPyro sampling model in which the common and per-pulsar processes share
        one set of Fourier coefficients on the 'red' basis.

        The prior covariance of the coefficients is the sum of the per-pulsar and
        common prior covariances, and all coefficients are sampled through a
        standardizing transform. Requires every process to live on the 'red' basis
        without inter-frequency correlations; see sampling_model_marginalized otherwise.

        Note all parameters assume a uniform prior. If 'log'-parameters are
        supplied, they also get a uniform prior (so the argument of the log
        gets a log-uniform prior). The exception are pulsar distance parameters
        used in some deterministic models which use a normal prior.

        If the user desires different priors, they can supply those in the
        'additional_ln_factor' input to the PTAModel object. See the advanced
        modeling examples.
        """

        # standard mode: user supplies per-pulsar processes and (optionally) a common process
        if self.mode == 'standard':

            # per-pulsar processes: hyper-parameters and prior covariance
            phi_cube = 0.
            for psr_model in self.psr_models:
                psr_params = self._sample_psr_params(psr_model)
                phi_cube = phi_cube + psr_model.get_phi_cube(psr_params, self.data.freqs)

            # common process: hyper-parameters and prior covariance
            if self.include_gwb_model:
                gwb_params = self._sample_gwb_params()
                phi_cube = phi_cube + self.gwb_model.get_phi_cube(gwb_params, self.data.freqs)
        
        # custom mode: user supplies single spectral model
        else:

            # sample hyper-parameters of custom spectral model
            spectral_params = _sample(name=self.spectral_model.name,
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
            det_params = _sample(self.det_model.name, dist.Uniform(self.det_model.param_mins,  
                                                                          self.det_model.param_maxs)) 
            if self.det_model.with_psr_params:  # if pulsar parameters are required
                psr_phases = _sample('psr_phases', dist.Uniform(0., 2. * jnp.pi).expand([self.npsrs]))
                standard_psr_dists = _sample('standard_psr_dists', dist.Normal().expand([self.npsrs]))
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
        z = z.astype(jnp.float32)

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


    def _sample_psr_params(self, psr_model):
        """Sample the (Npsrs, Nparam) hyper-parameters of a per-pulsar process."""
        psr_params = _sample(name=psr_model.name,
                                    fn=dist.Uniform(low=psr_model.param_mins,
                                                    high=psr_model.param_maxs,).expand([self.npsrs, psr_model.nparams_base]))

        # additional ln-factor for these parameters specified by user
        if psr_model.additional_ln_factor is not None:
            numpyro.factor(f'additional_ln_factor_{psr_model.name}', psr_model.additional_ln_factor(psr_params))
        return psr_params


    def _sample_gwb_params(self):
        """Sample the hyper-parameters of the common process."""
        gwb_params = _sample(name=self.gwb_model.name,
                                    fn=dist.Uniform(low=self.gwb_model.param_mins,
                                                    high=self.gwb_model.param_maxs))

        # additional ln-factor for GWB parameters specified by user
        if self.gwb_model.additional_ln_factor is not None:
            numpyro.factor('additional_ln_factor_gwb', self.gwb_model.additional_ln_factor(gwb_params))
        return gwb_params


    def _sample_det_coeffs(self):
        """Sample the deterministic model parameters and return its Fourier coefficients."""
        det_params = _sample(self.det_model.name, dist.Uniform(self.det_model.param_mins,
                                                                      self.det_model.param_maxs))
        if self.det_model.with_psr_params:
            psr_phases = _sample('psr_phases', dist.Uniform(0., 2. * jnp.pi).expand([self.npsrs]))
            standard_psr_dists = _sample('standard_psr_dists', dist.Normal().expand([self.npsrs]))
            psr_dists = numpyro.deterministic('psr_dists', self.data.psr_dists_measured + standard_psr_dists * self.data.psr_dists_std)
        else:
            psr_phases = None
            psr_dists = None

        # additional ln-factor for deterministic parameters specified by user
        if self.det_model.additional_ln_factor is not None:
            ln_factor_det = self.det_model.additional_ln_factor(det_params, psr_phases, psr_dists)
            numpyro.factor('additional_ln_factor_det', ln_factor_det)

        return self.det_model.get_coeffs_func(det_params, psr_phases, psr_dists)


    def stacked_bases(self):
        """
        The bases stacked in the marginalized sampling model, [F_1 | F_2 | ... | F_D]:
        every basis used by a per-pulsar or common process (in the Data object's
        order), then the deterministic basis 'det' if a deterministic model is given.

        Returns
        -------
        columns : dict
            Maps each basis name to its slice of the columns of the stacked basis.
        """
        used = {m.basis for m in self.psr_models}
        if self.include_gwb_model:
            used.add(self.gwb_model.basis)
        names = [b for b in self.data.basis_names if b in used]
        if self.include_det_model:
            names.append('det')
        columns, start = {}, 0
        for b in names:
            n = self.data.basis_ncomponents(b)
            columns[b] = slice(start, start + n)
            start += n
        return columns


    def _whitened_basis(self, rtol=1e-10):
        """
        Per-pulsar whitened compression used by sampling_model_marginalized,
        computed once in float64 and cached.

        The noise-weighted Gram matrix of the stacked basis (see stacked_bases)
        C = [F_1 | ... | F_D]^T N^{-1} [F_1 | ... | F_D] is factored as C = X^T X,
        with X of shape (k, n) and k the numerical rank of C. Because the residual
        inner products obey C y = [F_1 | ... | F_D]^T N^{-1} r for the whitened data y,
        the likelihood of any coefficients v on the stacked basis is
        -0.5 |y - X v|^2 + const. Working with (X, y) removes the large,
        parameter-independent data power from the likelihood, which float32
        cannot otherwise resolve when signals (e.g. DM variations) are loud.

        Returns
        -------
        Xs : array, shape (Npsrs, kmax, n)
            Whitened bases, zero-padded to the largest rank kmax.
        ys : array, shape (Npsrs, kmax)
            Whitened data, zero-padded.
        """
        if getattr(self, '_whitened_cache', None) is not None:
            return self._whitened_cache

        # The first call may come from inside a jit/pmap trace (e.g. NUTS with parallel
        # chains initializes the model under pmap). Build the cache eagerly regardless,
        # so it holds concrete arrays rather than tracers that would leak into later traces.
        with jax.ensure_compile_time_eval():
            self._whitened_cache = self._build_whitened_basis(rtol)
        return self._whitened_cache


    def _build_whitened_basis(self, rtol):
        """Compute the (Xs, ys) cached by _whitened_basis."""
        bases = list(self.stacked_bases())
        Xs, ys = [], []
        for name in self.psr_names:
            C = np.block([[self.data.gram(name, a, b) for b in bases] for a in bases])
            r = np.concatenate([self.data.basis_Nr(name, a) for a in bases])

            lam, U = np.linalg.eigh(0.5 * (C + C.T))
            keep = lam > rtol * lam[-1]
            lam, U = lam[keep], U[:, keep]
            Xs.append(np.sqrt(lam)[:, None] * U.T)
            ys.append((U.T @ r) / np.sqrt(lam))

        kmax = max(len(y) for y in ys)
        n = Xs[0].shape[1]
        X_pad = np.zeros((self.npsrs, kmax, n))
        y_pad = np.zeros((self.npsrs, kmax))
        for i, (X, y) in enumerate(zip(Xs, ys)):
            X_pad[i, :len(y)] = X
            y_pad[i, :len(y)] = y

        return (jnp.asarray(X_pad, dtype=jnp.float32),
                jnp.asarray(y_pad, dtype=jnp.float32))


    def sampling_model_marginalized(self):
        """
        NumPyro sampling model with analytically marginalized per-pulsar
        coefficients, and sampled common (GWB) coefficients.

        Per pulsar, the coefficients of every per-pulsar process (e.g. intrinsic
        noise on the 'red' basis F, DM variations on the 'dm' basis F_dm) are
        marginalized in the whitened compression of _whitened_basis. With
        M = I + X_B phi_B X_B^T, where X_B are the whitened columns of the bases of
        the per-pulsar processes and phi_B = blockdiag(phi_red, phi_dm, ...) (the
        processes on a basis adding their covariances), the marginalized
        log-likelihood is

            -0.5 d^T M^{-1} d - 0.5 ln det M + const,   d = y - X_D a_det + X_gwb a_gwb,

        which is evaluated stably in float32: M >= I, no data power is carried, and the
        Cholesky factor of M comes from batched_cholesky_identity_plus_gram, which stays
        accurate when loud signals give M a dynamic range beyond 1 / float32 epsilon.
        The GWB coefficients a_gwb live on the first 2 * gwb_model.nfreqs columns of
        the GWB's basis and are sampled through a standardizing transform conditional
        on the marginalized per-pulsar processes. Without a gwb_model, only the
        hyper-parameters are sampled and the log-likelihood is
        -0.5 y^T M^{-1} y - 0.5 ln det M.
        """

        # per-pulsar hyper-parameters and prior covariances, grouped by basis; on a
        # basis with Tspan='psr', each pulsar's spectrum uses its own frequencies
        psr_phis = {}
        for psr_model in self.psr_models:
            psr_params = self._sample_psr_params(psr_model)
            freqs_axis = 0 if psr_model.per_psr_freqs else None
            phi = vmap(psr_model.get_phi_func, in_axes=(0, freqs_axis))(psr_params, psr_model.freqs)
            psr_phis[psr_model.basis] = psr_phis[psr_model.basis] + phi if psr_model.basis in psr_phis else phi

        # GWB hyper-parameters
        if self.include_gwb_model:
            gwb_params = self._sample_gwb_params()

        # whitened bases and data, columns ordered as in stacked_bases
        Xs, ys = self._whitened_basis()
        columns = self.stacked_bases()

        if self.include_det_model:
            # whitened data with the deterministic signal subtracted
            a_det = self._sample_det_coeffs()
            with jax.default_matmul_precision('highest'):
                ys_eff = ys - vmap(jnp.dot)(Xs[:, :, columns['det']], a_det)
        else:
            ys_eff = ys

        if self.include_gwb_model:
            # draw whitened Fourier coefficients for the background
            gwb_na = 2 * self.gwb_model.nfreqs
            z = numpyro.sample(name='z', fn=dist.Normal().expand([self.npsrs, gwb_na]))

            # NumPyro adds a standard normal probability density for the line above
            # this is not in the posterior, so we need to subtract it manually
            numpyro.factor('ln_inverse_normal_correction', 0.5 * jnp.sum(z**2))
            z = z.astype(jnp.float32)

        # M = I + B B^T with B = X_B chol(phi_B), built from Cholesky factors of the prior
        # covariances so phi^{-1} is never formed: (Np, kmax, kmax)
        with jax.default_matmul_precision('highest'):
            XL = jnp.concatenate([vmap(lambda X, phi: X @ jnp.linalg.cholesky(phi))(Xs[:, :, columns[b]], phi)
                                  for b, phi in psr_phis.items()], axis=2)
        M_chols = batched_cholesky_identity_plus_gram(XL)
        M_ln_dets = 2 * jnp.sum(jnp.log(jnp.diagonal(M_chols, axis1=1, axis2=2)), axis=1)

        # whiten the data by M
        w = vmap(lambda L, y: jsl.solve_triangular(L, y, lower=True))(M_chols, ys_eff)  # (Np, kmax)

        if not self.include_gwb_model:
            # no common process: marginalized likelihood -0.5 y^T M^{-1} y - 0.5 ln det M
            with jax.default_matmul_precision('highest'):
                wMw = jnp.sum(w**2)
            numpyro.factor('lnlike', -0.5 * wMw - 0.5 * jnp.sum(M_ln_dets))
            return

        # whiten the GWB basis by M
        X_gwb = Xs[:, :, columns[self.gwb_model.basis]][:, :, :gwb_na]
        Q = vmap(lambda L, X: jsl.solve_triangular(L, X, lower=True))(M_chols, X_gwb)   # (Np, kmax, gwb_na)
        # these Gram products need true float32 matmuls; at the GPU default (TF32 on
        # Ampere) the ln-likelihood is off by O(1) and NUTS step sizes collapse
        with jax.default_matmul_precision('highest'):
            Ws = vmap(lambda A: A.T @ A)(Q)             # X_gwb^T M^{-1} X_gwb
            Vs = -vmap(lambda A, v: A.T @ v)(Q, w)      # -X_gwb^T M^{-1} y_eff
            wMw = jnp.sum(w**2)                         # y_eff^T M^{-1} y_eff

        # covariance of GWB Fourier coefficients
        gwb_phi_spec = self.gwb_model.get_phi_func(gwb_params, self.gwb_model.freqs)
        gwb_phi_spec = gwb_phi_spec[:gwb_na, :gwb_na]
        gwb_phi_spec_cho_factors = jsl.cho_factor(gwb_phi_spec, lower=True)
        gwb_phi_inv_spec = jsl.cho_solve((gwb_phi_spec_cho_factors[0], True),
                                        jnp.identity(gwb_phi_spec_cho_factors[0].shape[0], dtype=gwb_phi_spec.dtype))
        gwb_phi_spec_lndet = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(gwb_phi_spec)) / utils.renorm))
        gwb_phi_lndet = self.npsrs * gwb_phi_spec_lndet + gwb_na * self.gwb_model.lndet_correlation_matrix

        # standardizing transform: MAP GWB coefficients given the marginalized
        # pulsar noise and the det-subtracted data
        gwb_sigma_inv = Ws + gwb_phi_inv_spec[None, :, :]
        Sigma_inv_L = vmap(lambda x: jsl.cholesky(x, lower=True))(gwb_sigma_inv)
        y = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=0))(Sigma_inv_L, Vs)
        a_hat = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Sigma_inv_L, y)
        Lz = vmap(lambda x, y: jsl.solve_triangular(x, y, lower=True, trans=1))(Sigma_inv_L, z)
        a_gwb = a_hat + Lz
        numpyro.deterministic('a_gwb', a_gwb)

        # ln-determinant Jacobian of standardizing transform
        lndet_Jac = -jnp.sum(jnp.log(jnp.diagonal(Sigma_inv_L, axis1=1, axis2=2)))
        numpyro.factor('ln_Jac', lndet_Jac)

        # marginalized likelihood: -0.5 (w + Q a)^T (w + Q a) - 0.5 ln det M
        # (includes the deterministic signal through y_eff, up to a constant)
        ln_likelihood_val = -0.5 * jnp.sum(vmap(lambda a, W: jnp.dot(a, jnp.dot(W, a)))(a_gwb, Ws))
        ln_likelihood_val += jnp.sum(vmap(lambda a, V: jnp.dot(a, V))(a_gwb, Vs))
        ln_likelihood_val += -0.5 * wMw - 0.5 * jnp.sum(M_ln_dets)
        numpyro.factor('lnlike', ln_likelihood_val)

        # evaluate the GWB prior
        gwb_phi_inv = jnp.kron(self.gwb_model.inv_correlation_matrix, gwb_phi_inv_spec)
        ln_prior_val = -0.5 * jnp.dot(a_gwb.flatten(), jnp.dot(gwb_phi_inv, a_gwb.flatten()))
        ln_prior_val += -0.5 * gwb_phi_lndet
        numpyro.factor('lnprior', ln_prior_val)


    def get_param_names_and_shapes(self, alt_model=None):
        """
        Get the (sampled not observed) parameter names
        and shapes from a NumPyro sampling model.

        Parameters
        ----------
        alt_model : Callable
            Alternative NumPyro sampling model to get parameter names
            and shapes from. Defaults to None in which case the
            self.sampling_model is used.

        Returns
        -------
        shapes : dict
            A dictionary with parameter names as keys and corresponding
            shapes as keys.
        """
        
        if alt_model is None:
            sampling_model = self.sampling_model
        else:
            sampling_model = alt_model
        
        trace = handlers.trace(
            handlers.seed(sampling_model, jr.PRNGKey(0))
        ).get_trace()

        shapes = {
            name: site["value"].shape
            for name, site in trace.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
            }
        
        return shapes


    # former names of the sampling models
    sampling_model_no_freq_corrs = sampling_model_shared
    sampling_model_with_freq_corrs = sampling_model_marginalized
