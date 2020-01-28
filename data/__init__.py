import os
from pathlib import Path
from astropy.time import Time
import pandas as pd

data_dir = Path().absolute()

t_peak_mjd = Time(58603.87, format="mjd")
t_neutrino = Time("2019-10-01T20:09:18.17", format='isot', scale='utc')

bran_z = 0.0512

photometry_path = os.path.join(data_dir, "BranStark.dat")
photometry_data = pd.read_table(photometry_path, skiprows=4, sep="\s+")

meerkat_path = os.path.join(data_dir, "at2019dsg_MeerKAT.txt")
meerkat_data = pd.read_table(meerkat_path, sep="\s+")

radio_data_files = [x for x in os.listdir(data_dir) if "at2019dsg_20" in x]
vla_data = []
for x in radio_data_files:
    with open(os.path.join(data_dir, x), "r") as f:
        rawdate = (x.split("_")[1][:8])
        obs_date = Time("{0}-{1}-{2}T00:00:00.00".format(rawdate[:4], rawdate[4:6], rawdate[6:]), format='isot',
                        scale='utc')
        for line in f.readlines():
            line = line.replace("\n", "")
            vla_data.append(tuple([obs_date.mjd] + [float(x) for x in line.split(" ")]))

vla_data = pd.DataFrame(vla_data, columns=["mjd", "frequency", "flux", "flux_err"])

xray_path = os.path.join(data_dir, "bran_lx_Swift.dat")
xray_data = pd.read_table(xray_path, sep="\s+")

gamma_path = os.path.join(data_dir, "TDE_uls_FermiLAT")
gamma_data = pd.read_table(gamma_path, sep=",")

