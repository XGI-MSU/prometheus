'''
Functions to help build instances of DeterministicModel found
in deterministic_models.py. For example, the function which outputs
the timing delays induced by a continuous wave is found here.
'''

import jax.numpy as jnp
from jax import lax

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


def cw_delay_evolve_float64(toas, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function is NOT float32 compatible and is
    used primarily to test the accuracy of the float32 version.
    
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
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    # get pulsar time
    toas_copy = (toas - utils.tref)
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


def cw_delay_full_prior_float32(toas_shifted_scaled, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the CW timing delays in float32 across the extended parameter prior:
        log10_mc  in [7, 10],  log10_fgw in [-9, -7].

    Parameters
    ----------
    toas : array
        (npsrs, ntoas) shaped array. Raw TOAs in seconds; the tref shift is
        computed internally in the input dtype before casting to float32.
    psr_pos : array
        (npsrs, 3) shaped array of Cartesian unit vectors to each pulsar.
    source_params : array
        Shape (8,): log10(Mc/Msun), log10(fgw/Hz), cos(inc), psi,
        log10(h), cos(gwtheta), gwphi, phase0.
    psr_phases : array
        Shape (npsrs,). CW phase at each pulsar.
    psr_dists : array
        Shape (npsrs,). Distance to each pulsar [kpc].

    Returns
    -------
    res : array
        Shape (npsrs, ntoas). Timing delays in nanoseconds [ns].
        NaN where the binary has merged (fac1*t > 1).
    """
    f32 = jnp.float32


    psr_pos       = jnp.asarray(psr_pos,       dtype=jnp.float32)
    source_params = jnp.asarray(source_params,  dtype=jnp.float32)
    psr_phases    = jnp.asarray(psr_phases,    dtype=jnp.float32)
    psr_dists     = jnp.asarray(psr_dists,     dtype=jnp.float32)

    # unpack parameters
    log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta, gwphi, phase0 = source_params
    p_phases = psr_phases[:, None]

    # unit conversion (cw_renorm scaling)
    mc  = f32(10.0) ** log10_mc * f32(utils.Tsun * utils.cw_renorm)
    fgw = f32(10.0) ** log10_fgw / f32(utils.cw_renorm)
    w0  = f32(jnp.pi) * fgw

    gwtheta = jnp.arccos(cos_gwtheta)
    inc     = jnp.arccos(cos_inc)

    p_dists = psr_dists * f32(utils.kpc / utils.c * utils.cw_renorm)
    dist    = f32(2.0) * mc ** f32(5.0 / 3.0) * (f32(jnp.pi) * fgw) ** f32(2.0 / 3.0) / f32(10.0) ** log10_h

    # antenna pattern + geometry
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    L1minCosmu = (p_dists * (f32(1.0) - cosMu))[:, None]   # (npsrs, 1)

    # shared constants
    phase0_orb = phase0 / f32(2.0)
    mc53       = mc ** f32(5.0 / 3.0)
    fac1       = f32(256.0 / 5.0) * mc53 * w0 ** f32(8.0 / 3.0)
    phase_prefac = w0 / fac1
    amp_prefac   = mc53 / (dist * w0 ** f32(1.0 / 3.0)) / f32(utils.cw_renorm)
    xs = fac1 * toas_shifted_scaled        # (npsrs, ntoas)
    xL = fac1 * L1minCosmu       # (npsrs, 1)

    THRESH = f32(0.1)

    # Earth phase
    earth_exact  = f32(8.0 / 5.0) * phase_prefac * (f32(1.0) - (f32(1.0) - xs) ** f32(5.0 / 8.0))
    earth_taylor = phase_prefac * (xs
                                   + f32(3.0 / 16.0)     * xs ** 2
                                   + f32(11.0 / 128.0)   * xs ** 3
                                   + f32(209.0 / 4096.0) * xs ** 4
                                   + f32(5643.0 / 163840.0) * xs ** 5)
    phase = phase0_orb + jnp.where(xs < THRESH, earth_taylor, earth_exact)

    # Earth amplitude
    alpha = amp_prefac * (f32(1.0) - xs) ** f32(1.0 / 8.0)

    # Pulsar phase
    A    = f32(1.0) + xL            # (npsrs, 1)
    beta = xs / A                   # (npsrs, ntoas)

    A58  = A ** f32(5.0 / 8.0)
    psr_exact  = f32(8.0 / 5.0) * phase_prefac * A58 * (f32(1.0) - (f32(1.0) - beta) ** f32(5.0 / 8.0))
    psr_taylor = f32(8.0 / 5.0) * phase_prefac * A58 * (
        f32(5.0 / 8.0)        * beta
        + f32(15.0 / 128.0)   * beta ** 2
        + f32(55.0 / 1024.0)  * beta ** 3
        + f32(1045.0 / 32768.0)  * beta ** 4
        + f32(5643.0 / 262144.0) * beta ** 5)

    phase_p = phase0_orb + p_phases + jnp.where(beta < THRESH, psr_taylor, psr_exact)

    # Pulsar amplitude
    alpha_p = amp_prefac * (A - xs) ** f32(1.0 / 8.0)

    # waveform coefficients
    inc_factor = f32(-0.5) * (f32(3.0) + jnp.cos(f32(2.0) * inc))
    At   = jnp.sin(f32(2.0) * phase)   * inc_factor
    Bt   = f32(2.0) * jnp.cos(f32(2.0) * phase)   * cos_inc
    At_p = jnp.sin(f32(2.0) * phase_p) * inc_factor
    Bt_p = f32(2.0) * jnp.cos(f32(2.0) * phase_p) * cos_inc

    c2psi = jnp.cos(f32(2.0) * psi)
    s2psi = jnp.sin(f32(2.0) * psi)

    rplus    = alpha   * (-At   * c2psi + Bt   * s2psi)
    rcross   = alpha   * ( At   * s2psi + Bt   * c2psi)
    rplus_p  = alpha_p * (-At_p * c2psi + Bt_p * s2psi)
    rcross_p = alpha_p * ( At_p * s2psi + Bt_p * c2psi)

    res = fplus[:, None] * (rplus_p - rplus) + fcross[:, None] * (rcross_p - rcross)
    return res * f32(utils.renorm)


def cw_delay_evolve_float32(toas_shifted_scaled, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function is float32 compatible.

    Parameters
    ----------
    toas_shifted_scaled : array
        (npsrs, ntoas) shaped array. Values should be ``(toas - tref) * cw_renorm``
        (i.e. scaled by ``utils.cw_renorm = 1e-10``, not by ``utils.renorm``).
    psr_pos : array
        (npsrs, 3) shaped array of Cartesian unit vectors to each pulsar.
    source_params : array
        Shape (8,): log10(Mc/Msun), log10(fgw/Hz), cos(inc), psi,
        log10(h), cos(gwtheta), gwphi, phase0.
    psr_phases : array
        Shape (npsrs,). CW phase at each pulsar.
    psr_dists : array
        Shape (npsrs,). Distance to each pulsar [kpc].

    Returns
    -------
    res : array
        Shape (npsrs, ntoas). Timing delays in nanoseconds [ns].
    """
    f32 = jnp.float32

    psr_pos      = jnp.asarray(psr_pos,      dtype=jnp.float32)
    source_params = jnp.asarray(source_params, dtype=jnp.float32)
    psr_phases   = jnp.asarray(psr_phases,   dtype=jnp.float32)
    psr_dists    = jnp.asarray(psr_dists,    dtype=jnp.float32)

    # unpack source parameters
    log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta, gwphi, phase0 = source_params
    p_phases = psr_phases[:, None]

    # convert units, rescaled by cw_renorm so all intermediate values are O(1)
    # mc  [s * cw_renorm],  fgw [Hz / cw_renorm]
    mc   = f32(10.0) ** log10_mc * f32(utils.Tsun * utils.cw_renorm)
    fgw  = f32(10.0) ** log10_fgw / f32(utils.cw_renorm)
    w0   = f32(jnp.pi) * fgw

    gwtheta = jnp.arccos(cos_gwtheta)
    inc     = jnp.arccos(cos_inc)

    # pulsar distances
    p_dists = psr_dists * f32(utils.kpc / utils.c * utils.cw_renorm)

    # luminosity distance
    dist = f32(2.0) * mc ** f32(5.0 / 3.0) * (f32(jnp.pi) * fgw) ** f32(2.0 / 3.0) / f32(10.0) ** log10_h

    # antenna patterns
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    L1minCosmu = (p_dists * (f32(1.0) - cosMu))[:, None]   # (npsrs, 1)
    tp = toas_shifted_scaled - L1minCosmu                             # pulsar retarded time (npsrs, ntoas)

    # shared constants
    phase0_orb = phase0 / f32(2.0)
    mc53 = mc ** f32(5.0 / 3.0)
    fac1 = f32(256.0 / 5.0) * mc53 * w0 ** f32(8.0 / 3.0)
    phase_prefac = w0 / fac1

    # Earth phase
    x_e  = fac1 * toas_shifted_scaled
    phase = (phase0_orb
             + phase_prefac * (x_e
                               + f32(3.0 / 16.0)   * x_e ** 2
                               + f32(11.0 / 128.0) * x_e ** 3
                               + f32(209.0 / 4096.0) * x_e ** 4
                               + f32(5643.0 / 163840.0) * x_e ** 5))

    # Pulsar phase  (Taylor in xL = fac1*L and xt = fac1*tp)
    xL = fac1 * L1minCosmu        # (npsrs, 1)
    xt = fac1 * tp                # (npsrs, ntoas)
    xs = x_e                      # (npsrs, ntoas)
    xd = xL - xt                  # (npsrs, ntoas)

    xs2 = xs ** 2;  xd2 = xd ** 2
    phase_p = (phase0_orb
               + p_phases
               + phase_prefac * f32(8.0 / 5.0) * (
                   f32(5.0 / 8.0)        * xs
                   - f32(15.0 / 128.0)   * xs * xd
                   + f32(55.0 / 1024.0)  * xs * (xs2 + f32(3.0) * xd2) / f32(4.0)
                   - f32(1045.0 / 32768.0) * xs * xd * (xs2 + xd2) / f32(2.0)
                   + f32(5643.0 / 262144.0) * xs * (xs2 ** 2 + f32(10.0) * xs2 * xd2 + f32(5.0) * xd2 ** 2) / f32(16.0)))

    # Earth amplitude
    amp_prefac = mc53 / (dist * w0 ** f32(1.0 / 3.0)) / f32(utils.cw_renorm)
    alpha = amp_prefac * (f32(1.0)
                          - f32(1.0 / 8.0)      * x_e
                          - f32(7.0 / 128.0)    * x_e ** 2
                          - f32(35.0 / 1024.0)  * x_e ** 3
                          - f32(805.0 / 32768.0)  * x_e ** 4
                          - f32(4991.0 / 262144.0) * x_e ** 5)

    # Pulsar amplitude
    alpha_p = amp_prefac * (f32(1.0)
                            - f32(1.0 / 8.0)      * xt
                            - f32(7.0 / 128.0)    * xt ** 2
                            - f32(35.0 / 1024.0)  * xt ** 3
                            - f32(805.0 / 32768.0)  * xt ** 4
                            - f32(4991.0 / 262144.0) * xt ** 5)

    # waveform coefficients
    inc_factor = f32(-0.5) * (f32(3.0) + jnp.cos(f32(2.0) * inc))
    At   = jnp.sin(f32(2.0) * phase)   * inc_factor
    Bt   = f32(2.0) * jnp.cos(f32(2.0) * phase)   * cos_inc
    At_p = jnp.sin(f32(2.0) * phase_p) * inc_factor
    Bt_p = f32(2.0) * jnp.cos(f32(2.0) * phase_p) * cos_inc

    c2psi = jnp.cos(f32(2.0) * psi)
    s2psi = jnp.sin(f32(2.0) * psi)

    rplus    = alpha   * (-At   * c2psi + Bt   * s2psi)
    rcross   = alpha   * ( At   * s2psi + Bt   * c2psi)
    rplus_p  = alpha_p * (-At_p * c2psi + Bt_p * s2psi)
    rcross_p = alpha_p * ( At_p * s2psi + Bt_p * c2psi)

    res = fplus[:, None] * (rplus_p - rplus) + fcross[:, None] * (rcross_p - rcross)
    return res * f32(utils.renorm)

