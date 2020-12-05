import os
from pathlib import Path
from astropy.time import Time
from astropy import units as u
import astropy.io.ascii
import pandas as pd
import numpy as np

data_dir = Path(__file__).parent.absolute()

t_peak_mjd = Time(58603.87, format="mjd")
bran_disc = Time("2019-04-09T20:09:18.17", format='isot', scale='utc')
t_neutrino = Time("2019-10-01T20:09:18.17", format='isot', scale='utc')

bran_z = 0.0512

photometry_path = os.path.join(data_dir, "BranStark.dat")
photometry_data = pd.read_table(photometry_path, skiprows=4, sep="\s+")

marshal_path = os.path.join(data_dir, "bran_marshal.csv")
marshal_data = pd.read_csv(marshal_path)

radio_data_files = sorted([x for x in os.listdir(data_dir) if "at2019dsg_20" in x])[1:]
vla_data = []
for x in radio_data_files:
    with open(os.path.join(data_dir, x), "r") as f:
        rawdate = (x.split("_")[1][:8])
        obs_date = Time("{0}-{1}-{2}T00:00:00.00".format(rawdate[:4], rawdate[4:6], rawdate[6:]), format='isot',
                        scale='utc')
        for line in f.readlines():
            line = line.replace("\n", "")
            line = line.replace("#", "")
            vla_data.append(tuple([obs_date.mjd] + [float(x) for x in line.split(" ")]))

vla_data = pd.DataFrame(vla_data, columns=["mjd", "frequency", "flux", "flux_err"])
radio_data = vla_data.copy()
radio_data = radio_data.assign(instrument="VLA")


meerkat_path = os.path.join(data_dir, "at2019dsg_MeerKAT.txt")
meerkat_data = pd.read_table(meerkat_path, sep="\s+")
meerkat_data = meerkat_data.assign(
    mjd=meerkat_data["#mjd"],
    flux=meerkat_data["flux_mJy"],
    flux_err=meerkat_data["flux_err_mJy"],
    instrument="MeerKAT",
    frequency=1.4
)

ami_path = os.path.join(data_dir, "AT2019dsg_AMI.csv")
ami_data = pd.read_table(ami_path, sep=",")

mjds = []
for i, row in enumerate(ami_data.iterrows()):
    t = ami_data["Start Date"][i]
    x = t.split(" ")[0].split("/")
    date = "T".join(["-".join([x[2], x[0], x[1]]), t.split(" ")[1]])
    date = Time(date, format="isot")
    mjds.append(date.mjd)

ami_data = ami_data.assign(
    mjd=mjds,
    flux=ami_data["Peak"] * 1.e-3,
    flux_err=ami_data["Peak Error"] * 1.e-3,
    instrument="AMI-LA",
    frequency=15.5
)

merlin_path = os.path.join(data_dir, "at2019dsg_eMERLIN.txt")
merlin_data = pd.read_table(merlin_path, sep="\s+")
merlin_data = merlin_data.assign(
    mjd=Time(merlin_data["#date"][0], format="isot").mjd,
    instrument="eMERLIN",
    frequency=5.07
)

keys = ["mjd", "frequency", "flux", "flux_err", "instrument"]

radio_data = pd.concat(
    [
        radio_data,
        # merlin_data[keys],
        ami_data[keys],
        meerkat_data[keys]
    ],
    sort=True,
    ignore_index=True
)

radio_qs_epochs = None

for x in radio_data["mjd"]:
    if radio_qs_epochs is None:
        radio_qs_epochs = np.array([float(x)])

    else:

        mask = abs(radio_qs_epochs - x) < 2.

        if np.sum(mask) > 0:
            radio_qs_epochs = np.append(radio_qs_epochs, radio_qs_epochs[mask][0])
        else:
            radio_qs_epochs = np.append(radio_qs_epochs, x)


xrt_path = os.path.join(data_dir, "bran_lx_Swift.dat")
xrt_data = pd.read_table(xrt_path, sep="\s+")

xray_data = xrt_data.copy()
xray_data = xray_data.assign(instrument="XRT", UL=False, MJD=xray_data["#MJD"])

xrt_ul_path = os.path.join(data_dir, "bran_lx_Swift_ul.dat")
xrt_ul_data = pd.read_table(xrt_ul_path, sep="\s+")
xrt_ul_data = xrt_ul_data.assign(instrument="XRT", UL=True, MJD=xrt_ul_data["#MJD"])

xmm_path = os.path.join(data_dir, "at2019dsg_xmm.dat")
xmm_data = pd.read_table(xmm_path, sep="\s+")

xmm_ul = np.array([np.isnan(x) for x in xmm_data["tbb_kev"]])
xmm_mjd = [Time(f"{x}T00:00:00", format="isot").mjd for x in xmm_data["#date"]]

xmm_data = xmm_data.assign(instrument="XMM", UL=xmm_ul, MJD=xmm_mjd)

xray_data = pd.concat([
    xray_data,
    xrt_ul_data,
    xmm_data
],
    sort = True,
    ignore_index = True
)


gamma_path = os.path.join(data_dir, "TDE_uls_FermiLAT")
gamma_data = pd.read_table(gamma_path, sep=",")

gamma_data["UL(95)"] *= (1.* u.MeV).to("erg").value
gamma_deintegrate = np.log(800/0.1)


# SVV addition: 
radio_rec = astropy.io.ascii.read('./data/at2019dsg_merged.dat', format='fixed_width')

# force 5% callibration errors (no on MeerKAT because these have already been applied)
iincr = (radio_rec['inst']!='MeerKAT') * (radio_rec['eflux_mJy']>0)
radio_rec['eflux_mJy'][iincr] = np.sqrt(radio_rec[iincr]['eflux_mJy']**2+ (0.05*radio_rec[iincr]['flux_mJy'])**2) # add 5% errors




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

gfu_ehe_path = os.path.join(data_dir, "Aeff_ehe_gfu.csv")
aeff_ehe_gfu = pd.read_csv(gfu_ehe_path, names=["E_TeV", "A_eff"])

ehe_path = os.path.join(data_dir, "Aeff_ehe.csv")
aeff_ehe = pd.read_csv(ehe_path, names=["E_TeV", "A_eff"])

hese_path = os.path.join(data_dir, "Aeff_hese.csv")
aeff_hese = pd.read_csv(hese_path, names=["E_TeV", "A_eff"])

spectra_dir = os.path.join(data_dir, "ZTF19aapreis_spectra")
spectra_paths = sorted([os.path.join(spectra_dir, x) for x in os.listdir(spectra_dir) if "ZTF19aapreis" in x])