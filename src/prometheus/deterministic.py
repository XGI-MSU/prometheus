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


def cw_delay_evolve_low_freq_float32(toas, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function IS float32 compatible. This function
    is only accurate for low CW frequencies: (log10_f < -8.2).
    
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
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

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


def cw_delay_evolve_float32_TESTING(toas, psr_pos, source_params, psr_phases, psr_dists):
    
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function IS float32 compatible.

    This function is in TESTING. It's purpose is to work better in float32 precision
    at high frequencies.
    
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

    # convert units
    mc = 10.0 ** log10_mc * utils.Tsun * utils.cw_renorm
    fgw = 10.0 ** log10_fgw / utils.cw_renorm
    w0 = jnp.pi * fgw

    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)

    p_dists = psr_dists * utils.kpc / utils.c * utils.cw_renorm
    dist = 2.0 * mc ** (5.0 / 3.0) * (jnp.pi * fgw) ** (2.0 / 3.0) / 10.0 ** log10_h

    # antenna patterns
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    # times
    toas_copy = (toas - utils.tref) * utils.cw_renorm
    L1minCosmu = (p_dists * (1.0 - cosMu))[:, None]
    tp = toas_copy - L1minCosmu

    # constants
    phase0 = phase0 / 2.0
    mc53 = mc ** (5.0 / 3.0)
    w083 = w0 ** (8.0 / 3.0)
    fac1 = 256.0 / 5.0 * mc53 * w083

    # regime flags (scalar!)
    earth_use_exp = fac1 * jnp.max(toas_copy[:, -1]) < 0.1
    pulsar_use_exp = fac1 * jnp.maximum(
        tp[:, -1], p_dists * (1.0 - cosMu)
    ).max() < 0.1
    pulsar_amp_use_exp = fac1 * jnp.max(jnp.abs(tp[:, -1])) < 0.1

    # ----------------------
    # Earth phase
    # ----------------------
    def earth_phase_exp(_):
        x = fac1 * toas_copy
        return phase0 + w0 / fac1 * (
            x
            + 3/16 * x**2
            + 11/128 * x**3
            + 209/4096 * x**4
            + 5643/163840 * x**5
        )

    def earth_phase_orig(_):
        omega = w0 * (1.0 - fac1 * toas_copy) ** (-3.0 / 8.0)
        return phase0 + 1.0 / (32.0 * mc53) * (
            w0 ** (-5.0 / 3.0) - omega ** (-5.0 / 3.0)
        )

    phase = lax.cond(earth_use_exp, earth_phase_exp, earth_phase_orig, None)

    # ----------------------
    # Pulsar phase
    # ----------------------
    def pulsar_phase_exp(_):
        xL = fac1 * L1minCosmu
        xt = fac1 * tp
        return (
            phase0
            + p_phases
            + 8 * w0 / (5 * fac1) * (
                5/8 * xL + 5/8 * xt
                - 15/128 * xL**2 + 15/128 * xt**2
                + 55/1024 * xL**3 + 55/1024 * xt**3
                - 1045/32768 * xL**4 + 1045/32768 * xt**4
                + 5643/262144 * xL**5 + 5643/262144 * xt**5
            )
        )

    def pulsar_phase_orig(_):
        omega_p = w0 * (1.0 - fac1 * tp) ** (-3.0 / 8.0)
        omega_p0 = (
            w0 * (1.0 + fac1 * p_dists * (1.0 - cosMu)) ** (-3.0 / 8.0)
        )[:, None]
        return (
            phase0
            + p_phases
            + 1.0 / (32.0 * mc53)
            * (omega_p0 ** (-5.0 / 3.0) - omega_p ** (-5.0 / 3.0))
        )

    phase_p = lax.cond(pulsar_use_exp, pulsar_phase_exp, pulsar_phase_orig, None)

    # ----------------------
    # Earth amplitude
    # ----------------------
    def earth_alpha_exp(_):
        x = fac1 * toas_copy
        return (
            mc53
            / (dist * w0 ** (1.0 / 3.0))
            * (
                1
                - 1/8 * x
                - 7/128 * x**2
                - 35/1024 * x**3
                - 805/32768 * x**4
                - 4991/262144 * x**5
            )
            / utils.cw_renorm
        )

    def earth_alpha_orig(_):
        omega = w0 * (1.0 - fac1 * toas_copy) ** (-3.0 / 8.0)
        return mc53 / (dist * omega ** (1.0 / 3.0)) / utils.cw_renorm

    alpha = lax.cond(earth_use_exp, earth_alpha_exp, earth_alpha_orig, None)

    # ----------------------
    # Pulsar amplitude
    # ----------------------
    def pulsar_alpha_exp(_):
        x = fac1 * tp
        return (
            mc53
            / (dist * w0 ** (1.0 / 3.0))
            * (
                1
                - 1/8 * x
                - 7/128 * x**2
                - 35/1024 * x**3
                - 805/32768 * x**4
                - 4991/262144 * x**5
            )
            / utils.cw_renorm
        )

    def pulsar_alpha_orig(_):
        omega_p = w0 * (1.0 - fac1 * tp) ** (-3.0 / 8.0)
        return mc53 / (dist * omega_p ** (1.0 / 3.0)) / utils.cw_renorm

    alpha_p = lax.cond(
        pulsar_amp_use_exp, pulsar_alpha_exp, pulsar_alpha_orig, None
    )

    # ----------------------
    # Residuals
    # ----------------------
    inc_factor = -0.5 * (3.0 + jnp.cos(2.0 * inc))

    At = jnp.sin(2.0 * phase) * inc_factor
    Bt = 2.0 * jnp.cos(2.0 * phase) * cos_inc
    At_p = jnp.sin(2.0 * phase_p) * inc_factor
    Bt_p = 2.0 * jnp.cos(2.0 * phase_p) * cos_inc

    c2psi = jnp.cos(2.0 * psi)
    s2psi = jnp.sin(2.0 * psi)

    rplus = alpha * (-At * c2psi + Bt * s2psi)
    rcross = alpha * (At * s2psi + Bt * c2psi)
    rplus_p = alpha_p * (-At_p * c2psi + Bt_p * s2psi)
    rcross_p = alpha_p * (At_p * s2psi + Bt_p * c2psi)

    res = (
        fplus[:, None] * (rplus_p - rplus)
        + fcross[:, None] * (rcross_p - rcross)
    )

    return res * utils.renorm


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

