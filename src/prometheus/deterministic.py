'''
Functions to help build instances of DeterministicModel found
in deterministic_models.py. For example, the function which outputs
the timing delays induced by a continuous wave is found here.
'''

from jax.scipy.stats import norm
import jax.numpy as jnp
from . import utilities as utils


def get_psr_phase(cw_source_params, psr_position, psr_dist):
    """
    Compute the phase of a continuous gravitational wave at
    a pulsar given the GW parameters and pulsar position and distance.
    This is used for consistent injections, and not in the analysis.
    
    Parameters
    ----------
    cw_source_params : array
        Array of shape (8,) storing parameters of the CW source. We use the ordering:
        log10(chirp mass [solar mass]), log10(frequency [Hz]), cosine(inclination angle),
        polarization angle, log10(characteristic strain), cosine(polar angle of sky location),
        azimuthal angle of sky location, initial phase.
    psr_position : array
        Array of shape (3,) which is the Cartesian unit vector pointing to the pulsar.
    psr_dist : float
        Distance to pulsar in kpc.
    
    Returns
    -------
    psr_phase : float
        Phase of continuous gravitational wave at the pulsar.
    """

    # unpack CW parameters
    log10_mc, log10_fgw, cosinc, psi, log10_dist, costheta, gwphi, phase0 = cw_source_params

    # sky location / orientation
    singwtheta = jnp.sin(jnp.arccos(costheta))
    cosgwtheta = costheta
    singwphi = jnp.sin(gwphi)
    cosgwphi = jnp.cos(gwphi)
    omhat = jnp.array([-singwtheta * cosgwphi, -singwtheta * singwphi, -cosgwtheta])

    # compute pulsar phase
    cosMu = -jnp.dot(omhat, psr_position)
    pphase = (1 + 256/5 * (10**log10_mc*utils.Tsun)**(5/3) * (jnp.pi * 10**log10_fgw)**(8/3)
            * psr_dist*utils.Tkpc*(1-cosMu)) ** (5/8) - 1
    pphase /= 32 * (10**log10_mc*utils.Tsun)**(5/3) * (jnp.pi * 10**log10_fgw)**(5/3)
    psr_phase = -pphase%(2*jnp.pi)

    return psr_phase


def create_gw_antenna_pattern(gwtheta, gwphi, psr_pos):
    """
    Create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).

    Parameters
    ----------
    gwtheta : float
        Polar angle sky location of CW source.
    gwphi : float
        Azimuthal angle sky location of CW source.
    psr_pos : array
        (npsrs, 3) shaped array where npsrs is the number of pulsars in the array.
        These are the Cartesian unit vectors denoting the position of each pulsar.
    
    Returns
    -------
    (fplus, fcross, cosMu) : tuple
        fplus and fcross are the plus and cross antenna pattern functions, respectively,
        and are each arrays of shape (npsrs,) where npsrs are the number of pulsars
        in the array. cosMu is an array of shape (npsrs,) and the cosine of the angle
        between each pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    sgwphi = jnp.sin(gwphi)
    cgwphi = jnp.cos(gwphi)
    sgwtheta = jnp.sin(gwtheta)
    cgwtheta = jnp.cos(gwtheta)

    # this looks dumb, but it plays nice with JAX and batches across pulsars
    mdotpos = sgwphi * psr_pos[:, 0] - cgwphi * psr_pos[:, 1]
    ndotpos = -cgwtheta * cgwphi * psr_pos[:, 0] - cgwtheta * sgwphi * psr_pos[:, 1] \
                + sgwtheta * psr_pos[:, 2]
    omhatdotpos = -sgwtheta * cgwphi * psr_pos[:, 0] - sgwtheta * sgwphi * psr_pos[:, 1] \
                    -cgwtheta * psr_pos[:, 2]

    fplus = 0.5 * (mdotpos ** 2 - ndotpos ** 2) / (1 + omhatdotpos)
    fcross = (mdotpos * ndotpos) / (1 + omhatdotpos)
    cosMu = -omhatdotpos

    return fplus, fcross, cosMu


def cw_delay_evolve(toas, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function is NOT float32 compatible and is
    used primarily to test the accuracy of the float32 version below.
    
    Parameters
    ----------
    toas : array
        (npsrs, ntoas) shaped array where npsrs and ntoas are the number of pulsars
        and number of toas per pulsar respectively. Note these 'toas' need not be
        the actual observed TOAs of the array, and in implementation they are not.
        In deterministic models, the TOAs are a set of evenly spaced uniform TOAs
        used in the FFT. Then the Fourier design matrix maps the output of the
        FFT to the actual observed TOAs.
    psr_pos : array
        (npsrs, 3) shaped array where npsrs is the number of pulsars in the array.
        These are the Cartesian unit vectors pointing to each pulsar.
    source_params : array
        Array of shape (8,) storing parameters of the CW source. We use the ordering:
        log10(chirp mass [solar mass]), log10(frequency [Hz]), cosine(inclination angle),
        polarization angle, log10(characteristic strain), cosine(polar angle of sky location),
        azimuthal angle of sky location, initial phase.
    psr_phases : array
        Array of shape (npsrs,) where npsrs is the number of pulsars in the array.
        The array stores the phase of the CW at each pulsar.
    psr_dists : array
        Array of shape (npsrs,) where npsrs is the number of pulsars in the array.
        The array stores the distance to each pulsar [kpc].

    Returns
    -------
    res : array
        Array of same shape as 'toas' input. These are the delays in the timing residuals
        induced by the continuous wave in units of [ns].
    """
    # unpack parameters
    log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta, gwphi, phase0 = source_params
    p_phases = psr_phases
    pdists = psr_dists

    # convert units to time [s]
    mc = 10 ** log10_mc * utils.Tsun
    fgw = 10 ** log10_fgw
    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)
    p_dists = pdists * utils.kpc / utils.c
    dist = 2 * mc ** (5 / 3) * (jnp.pi * fgw) ** (2 / 3) / 10**log10_h

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    # get pulsar time
    toas_copy = toas - utils.tref
    tp = toas_copy - (p_dists*(1-cosMu))[:, None]

    # orbital frequency
    w0 = jnp.pi * fgw
    phase0 = phase0 / 2.0  # convert GW to orbital phase

    # calculate time dependent frequency at earth and pulsar
    mc53 = mc**(5./3.)
    w083 = w0**(8./3.)
    fac1 = 256./5. * mc53 * w083
    omega = w0 * (1. - fac1 * toas_copy)**(-3./8.)
    omega_p = w0 * (1. - fac1 * tp)**(-3./8.)
    omega_p0 = (w0 * (1. + fac1 * p_dists*(1-cosMu))**(-3./8.))[:, None]

    # calculate time dependent phase
    phase = phase0 + 1./32./mc53 * (w0**(-5./3.) - omega**(-5./3.))

    phase_p = (phase0 + p_phases[:, None]
                + 1./32./mc53 * (omega_p0**(-5./3.) - omega_p**(-5./3.)))

    # define time dependent coefficients
    inc_factor = -0.5 * (3. + jnp.cos(2. * inc))
    At = jnp.sin(2. * phase) * inc_factor
    Bt = 2. * jnp.cos(2. * phase) * cos_inc
    At_p = jnp.sin(2. * phase_p) * inc_factor
    Bt_p = 2. * jnp.cos(2. * phase_p) * cos_inc

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))

    # define rplus and rcross
    c2psi = jnp.cos(2. * psi)
    s2psi = jnp.sin(2. * psi)
    rplus = alpha*(-At*c2psi+Bt*s2psi)
    rcross = alpha*(At*s2psi+Bt*c2psi)
    rplus_p = alpha_p*(-At_p*c2psi+Bt_p*s2psi)
    rcross_p = alpha_p*(At_p*s2psi+Bt_p*c2psi)

    # residuals
    res = fplus[:, None] * (rplus_p - rplus) + fcross[:, None] * (rcross_p - rcross)
    return res * utils.renorm   # (Np, Nsparse)


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

def cw_delay_evolve_float32(toas, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function IS float32 compatible.
    
    Parameters
    ----------
    toas : array
        (npsrs, ntoas) shaped array where npsrs and ntoas are the number of pulsars
        and number of toas per pulsar respectively. Note these 'toas' need not be
        the actual observed TOAs of the array, and in implementation they are not.
        In deterministic models, the TOAs are a set of evenly spaced uniform TOAs
        used in the FFT. Then the Fourier design matrix maps the output of the
        FFT to the actual observed TOAs.
    psr_pos : array
        (npsrs, 3) shaped array where npsrs is the number of pulsars in the array.
        These are the Cartesian unit vectors pointing to each pulsar.
    source_params : array
        Array of shape (8,) storing parameters of the CW source. We use the ordering:
        log10(chirp mass [solar mass]), log10(frequency [Hz]), cosine(inclination angle),
        polarization angle, log10(characteristic strain), cosine(polar angle of sky location),
        azimuthal angle of sky location, initial phase.
    psr_phases : array
        Array of shape (npsrs,) where npsrs is the number of pulsars in the array.
        The array stores the phase of the CW at each pulsar.
    psr_dists : array
        Array of shape (npsrs,) where npsrs is the number of pulsars in the array.
        The array stores the distance to each pulsar [kpc].

    Returns
    -------
    res : array
        Array of same shape as 'toas' input. These are the delays in the timing residuals
        induced by the continuous wave in units of [ns].
    """
    # unpack parameters
    log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta, gwphi, phase0 = source_params
    p_phases = psr_phases[:, None]
    pdists = psr_dists

    # convert units to time [sec * cw_renorm]
    mc = 10 ** log10_mc * utils.Tsun * utils.cw_renorm
    fgw = 10 ** log10_fgw / utils.cw_renorm
    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)
    p_dists = (pdists * utils.kpc / utils.c * utils.cw_renorm)
    dist = 2 * mc ** (5 / 3) * (jnp.pi * fgw) ** (2 / 3) / 10**log10_h * utils.cw_renorm

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    # get pulsar time
    toas_copy = (toas - utils.tref) * utils.cw_renorm
    L1minCosmu = (p_dists * (1 - cosMu))[:, None]
    tp = toas_copy - L1minCosmu

    # orbital frequency
    w0 = jnp.pi * fgw
    phase0 = phase0 / 2.0  # convert GW to orbital phase

    # calculate time dependent frequency at earth and pulsar
    mc53 = mc**(5./3.)
    w083 = w0**(8./3.)
    fac1 = 256./5. * mc53 * w083

    # calculate time dependent phase
    phase = phase0 + w0 * toas_copy + 3./16. * w0 * fac1 * toas_copy**2 + 11./128.*toas_copy**3*w0*fac1**2
    phase_p = (phase0 + p_phases + w0 * (tp + L1minCosmu)
               + -3./16. * w0 * fac1 * (L1minCosmu**2 - tp**2)
               + 11./128.*w0*fac1**2*(L1minCosmu**3+tp**3))

    # define time dependent coefficients
    inc_factor = -0.5 * (3. + jnp.cos(2. * inc))
    At = jnp.sin(2. * phase) * inc_factor
    Bt = 2. * jnp.cos(2. * phase) * cos_inc
    At_p = jnp.sin(2. * phase_p) * inc_factor
    Bt_p = 2. * jnp.cos(2. * phase_p) * cos_inc

    # now define time dependent amplitudes
    alpha = mc53 / (dist * w0**(1./3.)) * (1 - 1./8.*fac1*toas_copy-7./128.*fac1**2*toas_copy**2)
    alpha_p = mc53 / (dist * w0**(1./3.)) * (1 - 1./8.*fac1*tp-7./128.*fac1**2*tp**2)

    # define rplus and rcross
    c2psi = jnp.cos(2. * psi)
    s2psi = jnp.sin(2. * psi)
    rplus = alpha*(-At*c2psi+Bt*s2psi)
    rcross = alpha*(At*s2psi+Bt*c2psi)
    rplus_p = alpha_p*(-At_p*c2psi+Bt_p*s2psi)
    rcross_p = alpha_p*(At_p*s2psi+Bt_p*c2psi)

    # residuals
    res = fplus[:, None] * (rplus_p - rplus) + fcross[:, None] * (rcross_p - rcross)
    return res * utils.renorm  # (Np, Nsparse)


# TODO: TEST THIS MODEL FOR SINGLE PRECISION AND CONVERGENCE WITH ENTERPRISE
# def cw_delay_monochromatic(toas, psr_pos, cw_source_params):
#     """
#     Returns the timing delays due to a non-evolving continuous
#     gravitational wave from an individual supermassive black
#     hole binary.
    
#     Parameters
#     ----------
#     toas : array
#         (npsrs, ntoas) shaped array where npsrs and ntoas are the number of pulsars
#         and number of toas per pulsar respectively. Note these toas need not be
#         the actual observed TOAs of the array, and in implementation they are not.
#         In deterministic models, the TOAs are a set of evenly spaced uniform TOAs
#         used in the FFT.
#     psr_pos : array
#         (npsrs, 3) shaped array where npsrs is the number of pulsars in the array.
#         These are the Cartesian unit vectors giving the position of each pulsar.
#     source_params : array
#         Array of shape (8,) storing parameters of the CW source. We use the ordering:
#         log10(chirp mass [solar masses]), log10(frequency [Hz]), cosine(inclination angle),
#         polarization angle, log10(characteristic strain), cosine(polar angle of sky location),
#         azimuthal angle of sky location, initial phase.
    
#     Returns
#     res : array
#         Array of same shape as 'toas' input. This is the delays in the timing residuals
#         induced by the continuous wave in units of ns.
#     -------
#     """

#     log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta, gwphi, phase0 = cw_source_params
    
#     # convert units to time
#     mc = 10 ** log10_mc * utils.Tsun
#     fgw = 10 ** log10_fgw
#     gwtheta = jnp.arccos(cos_gwtheta)
#     inc = jnp.arccos(cos_inc)
#     dist = 2 * mc ** (5 / 3) * (jnp.pi * fgw) ** (2 / 3) / 10**log10_h

#     # get antenna pattern funcs and cosMu
#     # write function to get pos from theta,phi
#     fplus, fcross, cosMu = create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

#     # get pulsar time
#     toas_copy = toas - utils.tref

#     # orbital frequency
#     phase0 = phase0 / 2  # convert GW to orbital phase

#     # monochromatic
#     omega = jnp.pi * fgw

#     # phases
#     phase = phase0 + omega * toas_copy

#     # define time dependent coefficients
#     At = -0.5 * jnp.sin(2 * phase) * (3 + jnp.cos(2 * inc))
#     Bt = 2 * jnp.cos(2 * phase) * jnp.cos(inc)

#     # now define time dependent amplitudes
#     alpha = mc ** (5.0 / 3.0) / (dist * omega ** (1.0 / 3.0))

#     # define rplus and rcross
#     rplus = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
#     rcross = alpha * (At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

#     # residuals
#     res = fplus[:, None] * rplus + fcross[:, None] * rcross
    
#     return -res * utils.renorm

