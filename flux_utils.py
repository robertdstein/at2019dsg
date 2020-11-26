import numpy as np
from astropy.coordinates import Distance
from astropy import units as u
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

flux_conversion =  (1+bran_z)/area

colors = {
    "r.IOO": "r",
    "r.ZTF": "r",
    "r.SEDm": "r",
    "g.ZTF": "g",
    "g.IOO": "g",
    "UVW2": "violet",
    "UVM2": "purple",
    "UVW1": "darkblue",
    "U": "lightblue",

}

bands = {
    "U": 3465 * u.angstrom,
    "UVW1": 2600 * u.angstrom,
    "UVM2": 2246 * u.angstrom,
    "UVW2": 1928 * u.angstrom,
    "g.ZTF": 464 * u.nm,
    "r.ZTF": 658 * u.nm,
    "g": 464 * u.nm,
    "r": 658 * u.nm,
    # "r.SEDm": 658 * u.nm,
}