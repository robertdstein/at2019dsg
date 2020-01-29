import numpy as np
from astropy.coordinates import Distance
from data import bran_z

def convert_radio(flux_mjy, frequency_ghz):
    flux_jy = 10**-3 * flux_mjy
    frequency_hz = 10**9 * frequency_ghz
    return 10**-23 * flux_jy * frequency_hz

def convert_to_mjy(energy_flux, frequency_ghz):
    frequency_hz = 10**9 * frequency_ghz
    return 10 ** 3 * 10 ** 23 * energy_flux / frequency_hz

dl = Distance(z=bran_z).to("cm").value
area = 4 * np.pi * (dl ** 2)

flux_conversion = 1./area