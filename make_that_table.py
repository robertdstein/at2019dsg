'''
make table with equiparition fitting results
and
table with all the radio data
'''
from data import bran_disc, radio_rec


import astropy.io.ascii
from numpy import *



dd = {}
# epsion_B = 0.001
# epsion_e = 0.1
# eta = epsion_B/epsion_e / (6/11)

# Eeq_corr = (11/17)*eta**(-6/17) + (6/17) *eta**(11/17)
# Req_corr = eta**(1/17)
# print (Req_corr, Eeq_corr)

# new: just the radio data
print ('\n')

data_rec = radio_rec
data_rec = data_rec[data_rec['inst']!='eMERLIN']

mjd0 = bran_disc.mjd
mjds = np.array([58625, 58653, 58703, 58761, 58819])

for i, mjd in enumerate(mjds): 
	
	it = abs(data_rec['mjd']-mjd)<14
	ii = it * (data_rec['eflux_mJy']>0)

	epoch_data  = data_rec[ii]
	for j in np.argsort(epoch_data ['nu_GHz']):
		print ('{0:6.0f}&{1:7.2f}&{2:5.3f}&{3:5.3f}&{4:9}\\\\'.format(epoch_data[j]['mjd']-mjd0, epoch_data[j]['nu_GHz'], epoch_data[j]['flux_mJy'], epoch_data[j]['eflux_mJy'], epoch_data[j]['inst']))
	print ('''\hline''')


print ('\n\n\n')
key = input('next table?')
print ('\n\n\n')

# read fit results
for eps_str in ['Eqp','nonEqp']:
	dd[eps_str] = astropy.io.ascii.read('./data/at2019dsg_mcmc_time_{0}.dat'.format(eps_str), format='fixed_width')

# print table 1:
for i, d1 in enumerate(dd['Eqp']):
	d01=dd['nonEqp'][i]
	ss = '{0:4} & ${1:5.2f}\pm{2:0.2f}$ & '.format((int(d1['time'])),10**d1['F_p'], np.log(10)*d1['eF_p']*10**d1['F_p'] )
	ss += '${0:5.1f}\pm{1:0.1f}$ \\\\ '.format(10**d1['nu_p']/1e9, np.log(10)*d1['enu_p']*10**d1['nu_p']/1e9 )
	print (ss)
print ('\n\n')

# print table 2:
for geo in ['cone', 'sphere']:
	print ('')
	for i, d1 in enumerate(dd['Eqp']):
		d01=dd['nonEqp'][i]
		ss = '&{0:4}& '.format((int(d1['time'])))
		ss += '${0:0.2f}({1:0.2f})$ & '.format(d1['R_'+geo], d1['eR_'+geo])
		ss += '${0:0.1f}({1:0.1f})$ &'.format(d1['E_'+geo], d1['eE_'+geo])
		ss += '${0:0.2f}({1:0.2f})$ & '.format(d1['B_'+geo], d1['eB_'+geo])
		ss += '${0:0.1f}({1:0.1f})$ &'.format(d1['n_electr_'+geo], d1['en_electr_'+geo])
		ss += '${0:0.2f}({1:0.2f})$ & '.format(d01['R_'+geo], d01['eR_'+geo])
		ss += '${0:0.1f}({1:0.1f})$ & '.format(d01['E_'+geo], d01['eE_'+geo])
		ss += '${0:0.2f}({1:0.2f})$ & '.format(d01['B_'+geo], d01['eB_'+geo])
		ss += '${0:0.1f}({1:0.1f})$ \\\\'.format(d01['n_electr_'+geo], d01['en_electr_'+geo])

		print (ss)





