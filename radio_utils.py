import numpy as np
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from data import bran_z, vla_data
from astropy import constants as const

# Formula from https://arxiv.org/abs/1510.01226

# f_a = 1.0 for spherical or 0.1 for conical

# Volume is shell of radius 0.1
f_v_shell = 4. / 3. * (1. ** 3 - 0.9 ** 3.)

# volume_of_cone = 4/3 pi r^3 times ratio of cone area to sphere surface area
# Volume of cone is thus f_v * f_a

# Works for spherical case now :)

# Equation 27 and 28 with p=3

# def equipartition_radius(peak_f_ghz, peak_flux_mjy, f_a=1.0, z=bran_z):
#     return (3.2 * 10 ** 15) * u.cm * (
#             (peak_flux_mjy ** (9. / 19.)) *
#             ((cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 26)) ** (18. / 19.)) *
#             ((peak_f_ghz / 10.) ** -1.) *
#             ((1 + z) ** (-10. / 19.)) *
#             (f_a ** (-8. / 19.)) *
#             ((f_v * f_a) ** (-1. / 19.))
#     ) * 4. ** (1. / 19.)

# Typo in formula!!! Should be -28/19 i.e -(19+9) not -19 + 9

def equipartition_radius(peak_f_ghz, peak_flux_mjy, f_a=1.0, z=bran_z):
    return (3.2 * 10 ** 15) * u.cm * (
            (peak_flux_mjy ** (9. / 19.)) *
            ((cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 26)) ** (18. / 19.)) *
            ((peak_f_ghz / 10.) ** -1.) *
            ((1 + z) ** (-28. / 19.)) *
            (f_a ** (-8. / 19.)) *
            ((f_v * f_a) ** (-1. / 19.))
    ) * 4. ** (1. / 19.)


def equipartition_energy(peak_f_ghz, peak_flux_mjy, f_a=1.0, z=bran_z):
    print((peak_flux_mjy ** (23. / 19.)),
          ((cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 26)) ** (46. / 19.)),
            ((peak_f_ghz / 10.) ** -1.),
            ((1 + z) ** (-42. / 19.)),
            (f_a ** (-12. / 19.)),
            ((f_v) ** (8. / 19.)))


    # return (1.9 * 10 ** 46) * u.erg * (
    #         (peak_flux_mjy ** (23. / 19.)) *
    #         ((cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 26)) ** (46. / 19.)) *
    #         ((peak_f_ghz / 10.) ** -1.) *
    #         ((1 + z) ** (-42. / 19.)) *
    #         (f_a ** (-12. / 19.)) *
    #         (f_v ** (8. / 19.))
    # ) * 4. ** (11. / 19.)



    return (1.9 * 10 ** 46) * u.erg * (
            (peak_flux_mjy ** (23. / 19.)) *
            ((cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 26)) ** (46. / 19.)) *
            ((peak_f_ghz / 10.) ** -1.) *
            ((1 + z) ** (-42. / 19.)) *
            (f_a ** (-12. / 19.)) *
            (f_v ** (8. / 19.))
    ) * 4. ** (11. / 19.)

def equipartition_beta(peak_f_ghz, peak_flux_mjy, delta_t, f_a=1.0, z=bran_z):
    r_eq = equipartition_radius(peak_f_ghz, peak_flux_mjy, f_a=f_a, z=z).to("cm").value
    f = (const.c * delta_t).to("cm").value/(r_eq * (1 + z))

    return 1./ (1. + f)

peak_f = []
peak_flux = []
dates = []
for date in sorted(list(set(vla_data["mjd"]))):
    data = vla_data[vla_data["mjd"] == date]
    max_index = list(data["flux"]).index(max(data["flux"]))
    peak_f.append(list(data["frequency"])[max_index])
    peak_flux.append(max(data["flux"]))
    dates.append(date)
dates = np.array(dates)

peak_f = np.array(peak_f)
peak_flux = np.array(peak_flux)

def get_delta_times(times=dates):
    return (times[1:] - times[:-1]) * u.day

def calc_av_expansion(f, flux, times=dates, z=bran_z, f_a=1):
    delta_t = get_delta_times(times)
    rads = equipartition_radius(f, flux, f_a=f_a, z=z)
    delta_rad = rads[1:] - rads[:-1]
    vel = (delta_rad / delta_t).to("m s-1")
    return vel

def equipartition_energy_duran(peak_f_ghz, peak_flux_mjy, gamma, f_v=f_v_shell, f_a=1.0, z=bran_z, epsilon_e=1., p=3.):

    chi_e = ((p - 2)/(p-1)) * epsilon_e * const.m_p / const.m_e

    print(21.8 ** ((-2 * (p + 1))/(13. + 2*p)),
        525 ** ((p - 1.) * 11./(13. + 2*p)) * chi_e ** ((2. - p) * 11./(13. + 2*p)),
        peak_flux_mjy ** ((14. + 3 * p)/(13 + 2 * p)),
        (cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 28)) ** ((2 * (3*p + 14.))/(13 + 2 * p)),
        ((peak_f_ghz / 10.) ** -1.),
        ((1 + z) ** (-(27. + 5 * p)/ (13 + 2 * p))),
        f_a ** (-3 * (p + 1)/(13 + 2*p)),
        f_v ** (2 * (p - 1)/(13 + 2 * p)),
        gamma ** (-(5. * p + 16.)/(13 + 2*p)),
        max((gamma - 1.),  2./chi_e) ** (-11 * (p - 2)/(13 + 2*p)))

    print("checker", 2./chi_e)

    return (1.3 * 10**48 * u.erg) * (
        21.8 ** ((-2 * (p + 1))/(13. + 2*p)) *
        525 ** ((p - 1.) * 11./(13. + 2*p)) *
        chi_e ** ((2. - p) * 11./(13. + 2*p)) *
        peak_flux_mjy ** ((14. + 3 * p)/(13 + 2 * p)) *
        (cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 28)) ** ((2 * (3*p + 14.))/(13 + 2 * p)) *
        ((peak_f_ghz / 10.) ** -1.) *
        ((1 + z) ** (-(27. + 5 * p)/ (13 + 2 * p))) *
        f_a ** (-3 * (p + 1)/(13 + 2*p)) *
        f_v ** (2 * (p - 1)/(13 + 2 * p)) *
        gamma ** -((5. * p + 16)/(13 + 2*p)) *
        max((gamma - 1.),  2./chi_e) ** (-11 * (p - 2)/(13 + 2*p))
    )  * 4. ** (11. / 19.)

def equipartition_energy_simple(peak_f_ghz, peak_flux_mjy, f_v=f_v_shell, f_a=1.0, z=bran_z, epsilon_e=1., p=3.):

    return (2.5 * 10**49 * u.erg *
        peak_flux_mjy ** (20. / 17.) *
        (cosmo.luminosity_distance(z=z).to(u.cm) / (u.cm * 10 ** 28)) ** (40./17.) *
        ((peak_f_ghz / 10.) ** -1.) *
        ((1 + z) ** (-25./17) )
            )