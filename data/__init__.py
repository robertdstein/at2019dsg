import os
from pathlib import Path
from astropy.time import Time
import pandas as pd
import numpy as np

data_dir = Path(__file__).parent.absolute()

t_peak_mjd = Time(58603.87, format="mjd")
bran_disc = Time("2019-04-09T20:09:18.17", format='isot', scale='utc')
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

xray_ul_path = os.path.join(data_dir, "bran_lx_Swift_ul.dat")
xray_ul_data = pd.read_table(xray_ul_path, sep="\s+")

gamma_path = os.path.join(data_dir, "TDE_uls_FermiLAT")
gamma_data = pd.read_table(gamma_path, sep=",")

# =================================
# ASASSN-14li data
# =================================

# ASASSN-14li data taken from Table 2 in https://arxiv.org/abs/1510.01226

asassn_14li_data = np.array([
    [143., 8.20, 1.76],
    [207., 4.37, 1.23],
    [246., 4.00, 1.14],
    [304., 2.55, 0.94],
    [381., 1.91, 0.62]
])

# ASASSN-14li t0 here is extrapolated outflow launch date, on

asassn_14li_t0 = Time("2014-08-18T00:00:00.00", format='isot', scale='utc')
asassn_14li_disc = Time("2014-11-22T00:00:00.00", format='isot', scale='utc')

asassn_14li_z = 0.0206

# =================================
# XMMSL1 J0740−85 data
# =================================

# ASASSN-14li data taken from Table 1 in https://arxiv.org/abs/1610.03861

xmmsl1_data = np.array([
    [609., 1.5, 1.19],
    [769., 1.7, 0.89],
    [875., 1.6, 1.6]
])

xmmsl1_z = 0.0173