import numpy as np
from astropy import constants as const
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u

def plancks_law(nu_hz, t):
    nu = (nu_hz * u.Hz).to("s^-1")
    temp = t * u.K
    ratio = (const.h * nu) / (const.k_B * temp)
    return (2. * const.h * nu ** 3.) / (const.c ** 2. * (np.exp(ratio) - 1.)) / u.sr

def plancks_law_wl(wl_m, t):
    wl = (wl_m * u.m).to("nm")
    temp = t * u.K
    ratio = (const.h * const.c) / (const.k_B * temp * wl)
    res = (2. * const.h * const.c ** 2.) / (wl**5. * (np.exp(ratio) - 1.)) / u.sr
    return res.to("kW sr^-1 m^-2 nm^-1")

def spectral_irradiance(nu_hz, t):
    return np.pi * u.sr * plancks_law(nu_hz, t)

def spectral_flux_density(nu_hz_obs, t, z, r_bb_cm):
    nu_hz_emit = (1 + z) * nu_hz_obs
    source_flux = spectral_irradiance(nu_hz_emit, t)
    dist = cosmo.luminosity_distance(z=z).to(u.cm)
    obs_flux = source_flux * (r_bb_cm * u.cm)/dist
    return obs_flux.to("erg cm^-2 s^-1 Hz^-1")