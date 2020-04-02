import astropy.io.ascii
from numpy import *

dd = {}
epsion_B = 0.001
epsion_e = 0.1
eta = epsion_B/epsion_e / (6/11)

Eeq_corr = (11/17)*eta**(-6/17) + (6/17) *eta**(11/17)
Req_corr = eta**(1/17)
print (Req_corr, Eeq_corr)


# read fit results
for eps_str in ['equip','eps_e0.100']:
	dd[eps_str] = astropy.io.ascii.read('./data/at2019dsg_mcmc_time_{0}.dat'.format(eps_str), format='fixed_width')

# print table:
for geo in ['cone', 'sphere']:
	print ('')
	for i, d1 in enumerate(dd['equip']):
		d01=dd['eps_e0.100'][i]
		ss = '&{0:4}&${1:5.2f}\pm{2:0.2f}$ & '.format((int(d1['time'])),10**d1['F_p'], np.log(10)*d1['eF_p']*10**d1['F_p'] )
		ss += '${0:5.1f}\pm{1:0.1f}$ & '.format(10**d1['nu_p']/1e9, np.log(10)*d1['enu_p']*10**d1['nu_p']/1e9 )
		ss += '${0:0.2f}\pm{1:0.2f}$ & '.format(d1['R_'+geo], d1['eR_'+geo])
		ss += '${0:0.1f}\pm{1:0.1f}$ &'.format(d1['E_'+geo], d1['eE_'+geo])
		ss += '${0:0.2f}\pm{1:0.2f}$ & '.format(d01['R_'+geo]+log10(Req_corr), d01['eR_'+geo])
		ss += '${0:0.1f}\pm{1:0.1f}$ \\\\'.format(d01['E_'+geo]+log10(Eeq_corr), d01['eE_'+geo])

		print (ss)

