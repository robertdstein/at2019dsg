import astropy.io.ascii
from scipy.optimize import leastsq

# personal import
from sjoert import sync

# local import
import equipartition_functions 
from importlib import reload
#reload(equipartition_functions)
from equipartition_functions import *

import emcee
import corner

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

credit_int = 68.2689492137086

nwalkers = 50
nsteps = 1500
burnin = 1000

silent = False
wait = False
plotname = './plots/bran'

z = equipartition_functions.z
D_L = sjoert.stellar.lumdis(z, h=0.7)


# read the radio data
data_rec = astropy.io.ascii.read('./data/at2019dsg_merged.dat', format='fixed_width')

#data_rec['eflux_mJy'] = np.clip(data_rec['eflux_mJy'], 0.05*data_rec['flux_mJy'], 1e99) # force errors?

# ---
#  input by hand the dates we want to fit the radius and B-field. 
mjd_fit = np.array([58625, 58653, 58703, 58761]) #58819 last epoch has only MeerKAT and AMI data
#mjd_fit = np.array([58761])# last epoch only (gives p=3)


# remove data that's not close enough to MJDs we want to fit for
dfit = np.array([min(abs(d['mjd']-mjd_fit)) for d in data_rec])
data_rec = data_rec[dfit<14]
data_rec = data_rec[data_rec['inst']!='eMERLIN']
#data_rec['eflux_mJy'][data_rec['inst']=='AMI'] *= 10 #data_rec

try:
	dummy = R_single[:,0]
except NameError:
	print ('First tine fit_radio_data to get least least-square results')
# ---
# free paremeters with our best guess (from least-square fit)
R_arr = R_single[:,0]
B_arr = B_single[:,0]
R_arr = np.log10(R_arr)
B_arr = np.log10(B_arr)
p_electron = 3.0										# electorn power-law index
Fbase0 = 0.1 											# baseline flux in mJy @1.4 GHz
Fbase1 = -1 											# spectral index of baseline flux
lnf = -2 												# fudge factor for errors

# fix parameter:
phi0 = pi/4.
phi0 = 30/180*pi
epsilon_e = 0.01

prior_dict={}
#prior_dict["p_electron"] =  {"min":2.2,"max":4.00,"sigma":None,"value":p_electron}
prior_dict["p_electron"] =  {"min":2.1,"max":4.00,"sigma":0.15,"value":3.0}
prior_dict["Fbase0"] =  {"min":0,"max":0.12,"sigma":None,"value":Fbase0}
prior_dict["Fbase1"] =  {"min":-2,"max":0,"sigma":None,"value":Fbase1}
prior_dict["lnf"] =    {"min":-5,"max":-2,"sigma":None,"value":-3}

# make sure the B-field doesn't run out of bounds of the interpolator
for i in range(len(R_arr)):
    prior_dict["B"+str(int(mjd_fit[i]-mjd_fit[0]))] =  {"min":np.log10(min(BB)),"max":np.log10(max(BB)),"sigma":None,"value":B_arr[i]}
    prior_dict["R"+str(int(mjd_fit[i]-mjd_fit[0]))] =  {"min":15.5,"max":17.5,"sigma":None,"value":R_arr[i]}



# define the ordering of the parameters 
par_names  = ["R"+str(int(mjd_fit[i]-mjd_fit[0])) for i in range(len(R_arr))]
par_names += ["B"+str(int(mjd_fit[i]-mjd_fit[0])) for i in range(len(R_arr))]
par_names += ["p_electron","Fbase0", "Fbase1", "lnf"]


def model_func(p, mjd, nu, verbose=False):
	
	R = 10**np.interp(mjd, mjd_fit, p[0:len(mjd_fit)])
	B = np.interp(mjd, mjd_fit, p[len(mjd_fit):len(mjd_fit)*2])	
	p_electron = np.repeat( np.log10(p[len(mjd_fit)*2]), len(mjd))
	Fbase0 = p[len(mjd_fit)*2+1]
	Fbase1 = p[len(mjd_fit)*2+2]
	if verbose:
		print ('log10(B)', B)
		print ('log10(R)', np.log10(R))
		print ('p_electr', p[len(mjd_fit)*2])

	# do synchtron prediction
	#mout = np.zeros(len(mjd))
	#for i in range(len(mjd)):
	mout = model_single_tab(nu, B, R, p_electron)	
	
	# convert to mJy
	mout *= 1e23*1e3 
	if verbose:
		print ('model single outout in mJy:', mout)

	# add baseline flux
	mout += Fbase0*(10**nu/1.4e9)**(Fbase1)

	#print (mout)

	return mout

# fit normalization for parameters with Gaussian prior (likely none in this project)
for par in par_names:
    if prior_dict[par].get("sigma") is not None:
        # if not silent:
        #     print('{2:7} prior {0:0.2f} +/- {1:0.3f}'.format(prior_dict[par]["value"], prior_dict[par]["sigma"], par))        
        prior_dict[par]["norm"] = -np.log(prior_dict[par]["sigma"])
        prior_dict[par]["ivar"] = 1/prior_dict[par]["sigma"]**2

# Set up the sampler.
ndim= len(par_names)
#guess_pos = result["x"]
#guess_pos = [norm_ls, np.log10(50), 0+tp_prior_max/2.,  -5/3., np.log(0.1)]
guess_pos = [prior_dict[par]["value"] for par in par_names]

par_with_gauss_prior = [par for par in par_names if prior_dict[par]['sigma']]
print ('parameters with gaussian prior', par_with_gauss_prior)

#check limits
for i, par in enumerate(par_names):
    if (guess_pos[i]<prior_dict[par]["min"]) or (guess_pos[i]>prior_dict[par]["max"]):
        print ('WARNING!! the start position ({0:0.3f}) is outside the limits of this prior:'.format(guess_pos[i]), prior_dict[par])




# Define the prior function     
def lnprior(theta):
    
    #check limits
    for i, par in enumerate(par_names):
        if theta[i]<prior_dict[par]["min"]:
            return -np.inf
        if theta[i]>prior_dict[par]["max"]:
            return -np.inf    

    # apply Gausian priors (if any)
    out = 0
    for i, par in enumerate(par_names):
        if par in par_with_gauss_prior:
            out += -0.5*(theta[i]-prior_dict[par]["value"])**2 * prior_dict[par]["ivar"] + prior_dict[par]["norm"]
    return out


# Define the likelihood, assuming Gaussian errors with fudge factor in the variance
def lnlike(theta, mjd, y, yerr, nu):
    
    # get a power-law
    model_y = model_func(theta[0:-1], mjd, nu)

    lnf = theta[-1]
    # print ('pars:',theta)
    # print ('model:', model_y)
    # key = input()
    inv_sigma2 = 1.0/(yerr**2 + model_y**2*np.exp(2*lnf))
    out = -0.5*(np.sum((y-model_y)**2*inv_sigma2 - np.log(inv_sigma2)))
    return out

# Define the probability function as likelihood * prior.
def lnprob(theta, x, y, yerr, nu):
    
    lp = lnprior(theta)
    
    # print ('pars:',theta)
    # print ('prior:', lp)
    
    # for i, par in enumerate(par_names):
    #     print (par,theta[i], prior_dict[par]["min"], prior_dict[par]["max"])
    #     print (theta[i]<prior_dict[par]["min"])
    #     print (theta[i]>prior_dict[par]["max"])

    # key = input()

    if not np.isfinite(lp):
        return -np.inf
    out = lp + lnlike(theta, x, y, yerr, nu)
    return out


guess_pos = [prior_dict[par]["value"] for par in par_names]
    
if not(silent):
    print("MCMC start:")
    yo = [print("{0:7}  = {1:0.2f}".format(par_names[l], guess_pos[l])) for l in range(ndim)]

# add small amount of scatter to start of walkers
pos = [guess_pos + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

# let's go
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
		args=(data_rec['mjd'], data_rec['flux_mJy'], data_rec['eflux_mJy'], np.log10(data_rec['nu_GHz']*1e9*(1+z))))

# Clear and run the production chain.    
sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state(), progress=True)



par_names_walk = par_names
if plotname:        

    fig, axes = plt.subplots(len(par_names_walk), 1, sharex=True, figsize=(10.4, 11.7))

    for l, par in enumerate(par_names_walk):
        axes[l].plot(sampler.chain[:, :, l].T, color="k", alpha=0.4)
        axes[l].yaxis.set_major_locator(MaxNLocator(5))
        axes[l].set_ylabel(par)

    axes[l].set_xlabel("step number")

    fig.tight_layout(h_pad=0.0)
    fig.savefig(plotname+"-mcmc-walkers.png")
    if wait:
        key = input()
    plt.close()

# Make the triangle plot.
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# dump the samples
import pickle
pickle.dump(samples, open('data/mcmc_samples.pickle','wb'))

if plotname:        

	fig = corner.corner(samples, labels=par_names)
	fig.savefig(plotname+"-mcmc-triangle.pdf")
	if wait:
	    key = input()
	plt.close()


# Compute the quantiles.
out_tup = [np.array([v[1], v[1]-v[0], v[2]-v[1]]) for v in zip(*np.percentile(samples, [50-credit_int/2., 50, 50+credit_int/2.], axis=0))] #[16, 50, 84]
    
out_dict = {par:tuple(out_tup[l]) for l, par in enumerate(par_names)}
bf_arr = np.array([out_tup[l][0] for l, par in enumerate(par_names[0:-1])])

if not(silent):
    print("MCMC result:")
    yo = [print("{0:10} = {1[0]:0.3f} -{1[1]:0.3f} +{1[2]:0.3f}".format(par, out_dict[par])) for par in par_names]
    print ("")

# dump mcmc percentiles result
import json
json.dump(out_dict, open('./data/at2019dsg_mcmc.json', 'w'), indent=3)

# --- 
# look over sample at each MJD and 
# make nice SED plot
plt.clf()
from cycler import cycler
cmap = mpl.cm.get_cmap('gist_heat')
cmap = mpl.cm.get_cmap('plasma')
cmap = mpl.cm.get_cmap('viridis')
custom_cycler = (cycler(color=([cmap(i) for i in np.linspace(0.6,0.0,len(mjd_fit))])))
plt.gca().set_prop_cycle(custom_cycler)

Nlc = 200
mjd0 = mjd_fit[0]-12
mjd0 = 58584 # first detection?

samples_dict = {k:np.zeros((len(mjd_fit), Nlc)) for k in \
 	('F_p', 'nu_p', 'R_SJ', 'R_cone', 'R_sphere', 'E_SJ','E_cone','E_sphere', 'B_SJ',  'n_electr','N_electr','E_electr', 'v_SJ', 'v_cone', 'v_sphere')}

for i, mjd in enumerate(mjd_fit): 
	
	it = abs(data_rec['mjd']-mjd)<10
	ii = it * (data_rec['eflux_mJy']>0)

	if sum(ii)>1:
		nu = data_rec[ii]['nu_GHz']
		Fnu = data_rec[ii]['flux_mJy'] 
		Fnu_err = data_rec[ii]['eflux_mJy'] 
		base = out_dict['Fbase0'][0]*(nu/1.4)**(out_dict['Fbase1'][0]) # subtract baseline

		lbl = sjoert.simtime.mjdtodate(np.mean(data_rec['mjd'][it])).strftime('%y/%m/%d')
		line = plt.errorbar(nu, Fnu,Fnu_err, fmt='o', label=lbl, zorder=10-i, alpha=0.8)
		# plt.pause(0.1)
		# key = input()

		xx = np.logspace(9,10.3, 1000)
		base_xx = out_dict['Fbase0'][0]*(xx/1.4e9)**(out_dict['Fbase1'][0]) # subtract baseline

		bf_func = model_func(bf_arr, np.repeat(mjd, len(xx)), np.log10(xx),verbose=False) #-base_xx
		plt.plot(xx/1e9, bf_func,  '--',alpha=0.7, color=line[0].get_color()) #label=ll1
		xmax = xx[np.argmax(bf_func)]
		plt.annotate('{0:0} d'.format(mjd-mjd0), (xmax/1e9/1.3, max(bf_func)/1.45), color=line[0].get_color())

		for l, parms in enumerate((samples[np.random.randint(len(samples), size=Nlc)])):

			# check for rough walkers
			df = np.interp(np.log10(5e9), np.log10(xx), bf_func) / model_func(parms[0:-1], np.array([mjd]), np.array([np.log10(5e9)]))

			if abs(np.log10(df))<0.3:
				
				this_base = parms[len(mjd_fit)*2+1]* (xx/1.4e9)**parms[len(mjd_fit)*2+2]
				this_model = model_func(parms[0:-1], np.repeat(mjd, len(xx)), np.log10(xx))
				plt.plot(xx/1e9, this_model,  '-',alpha=0.02, color=line[0].get_color()) #label=ll1

				imax = np.argmax(this_model-this_base)
				samples_dict['F_p'][i,l] = this_model[imax]-this_base[imax] #mJy, baseline subtracted
				samples_dict['nu_p'][i,l] = xx[imax]

				this_B = 10**parms[len(mjd_fit)+i]
				this_R = 10**parms[i]
				this_p = parms[len(mjd_fit)*2]
				#this_p = np.random.normal(3, 0.15) # fixed a last epoch measurement
				
				samples_dict['R_SJ'][i,l] = this_R
				samples_dict['B_SJ'][i,l] = this_B

 				# two cones
				D_cone = this_R/np.tan(phi0)
				#fA = (1-cos(phi0))/(1-cos(1))
				fA = 2 * 2*pi*(1-cos(phi0))/(4*pi)  # seems that 4*pi is need for the Newtonian case of Barniol equations
				if phi0<pi/4:
					fV = 2 * 1/3.*pi* D_cone*(this_R)**2 / (pi*this_R**3) # appropiate for small phi (<45 deg?)
				else:
					fV = 4/3*fA
				
				#fV *= 1-0.9**3 # emitting region is with in 0.1 of total length

				who = 'Barniol'				
				samples_dict['R_cone'][i,l] = sync.Req(this_model[imax], D_L, xx[imax], z, fA=fA, fV=fV, epsilon_e=epsilon_e, p=this_p, who=who)
				samples_dict['E_cone'][i,l] = sync.Eeq(this_model[imax], D_L, xx[imax], z, fA=fA, fV=fV, epsilon_e=epsilon_e, p=this_p, who=who)

				# sphere
				fA = 1 # one in newtonian case...
				fV = 4/3.

				samples_dict['R_sphere'][i,l] = sync.Req(this_model[imax], D_L, xx[imax], z, fA=fA, fV=fV, epsilon_e=epsilon_e, p=this_p, who=who)
				samples_dict['E_sphere'][i,l] = sync.Eeq(this_model[imax], D_L, xx[imax], z, fA=fA, fV=fV, epsilon_e=epsilon_e, p=this_p, who=who)
				
				samples_dict['E_SJ'][i,l] =  4/3*pi*(this_R)**3 * (this_B)**2/(8*np.pi) 
				
				samples_dict['n_electr'][i,l] = 1/(1+1/epsilon_e) * sync.K(this_B, this_p) / (this_p-1)
				samples_dict['N_electr'][i,l] = samples_dict['n_electr'][i,l] * (this_R**3) * 4/3.*pi
				samples_dict['E_electr'][i,l] = sync.me * sync.c**2 * sync.K(this_B, this_p, eps_e=equipartition_functions.eps_e) / (this_p-2) * (this_R**3) * 4/3.*pi
			else:
				print ('rejecting sample')

# get the velocities
for i, mjd in enumerate(mjd_fit[0:-1]): 
	for k in ('SJ', 'cone', 'sphere'):
		samples_dict['v_'+k][i, :] = (samples_dict['R_'+k][i, :]-samples_dict['R_'+k][i+1, :]) / ((mjd_fit[i]-mjd_fit[i+1])*3600*24) /3e10 


plt.xlim(1.0,20) 
plt.ylim(0.05, 2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux (mJy)')
plt.savefig('./plots/at2019dsg_radio.pdf')

# also write the best fit and uncertainty as a function of time
bf_all_cols = ['time']+list(samples_dict.keys()) + ['e'+x for x in samples_dict.keys()]
bf_dict = {k:0. for k in  bf_all_cols}
bf_rec = sjoert.rec.dict2rec(bf_dict, n=len(mjd_fit))

for i, mjd in enumerate(mjd_fit): 
	#dk = str(int(mjd_fit[i]-mjd_fit[0]))
	for k in samples_dict.keys():
		iok = samples_dict[k][i, :]!=0
		med, sig = np.log10(np.median(samples_dict[k][i, iok])), np.std(np.log10(samples_dict[k][i,iok]))
		# do linear for v
		if k[0] =='v':
			med, sig = np.median(samples_dict[k][i, iok]), np.std(samples_dict[k][i,iok])
		bf_rec[k][i], bf_rec['e'+k][i] = med, sig

for k in ('SJ', 'cone', 'sphere'):	
	bf_rec['v_'+k][-1] = 0
	bf_rec['ev_'+k][-1] = 0

bf_rec['time'] = mjd_fit - mjd0

out_cols = \
['time',
 'F_p',
 'eF_p',
 'nu_p',
 'enu_p',
 'R_SJ', 
 'R_sphere',
 'R_cone',
 'eR_SJ',
 'eR_cone',
 'E_SJ',
 'E_sphere',
  'E_cone',
 'eE_SJ',
 'eE_cone',
 'v_SJ', 
 'v_sphere', 
 'v_cone',
 'ev_SJ',
 'ev_cone']

astropy.io.ascii.write([bf_rec[x] for x in out_cols], './data/at2019dsg_mcmc_time.dat', 
		format='fixed_width', 
		names = out_cols,
		formats={k:'0.3f' for k in bf_dict.keys()},
		overwrite=True)

plt.pause(0.1)
key = input('next?')

# plt.clf()
# for i, mjd in enumerate(mjd_fit): 
# 	med, sig = np.log10(np.median(E_electron[i, :])), np.std(np.log10(E_electron[i,:]))
# 	plt.errorbar(mjd-mjd_fit[0]+12,med, sig,fmt='s')
# 	#out_dict['E_electr'+str(int(mjd_fit[i]-mjd_fit[0]))] = np.array([med, sig])
# plt.ylabel('Electron total energy (erg)')
# plt.pause(0.1)
# key = input('next?)')



plt.clf()
plt.errorbar(bf_rec['R_cone'],bf_rec['n_electr'], xerr=bf_rec['eR_cone'], yerr=bf_rec['en_electr'], fmt='s')
xx = np.linspace(min(bf_rec['R_cone'])-0.1, max(bf_rec['R_cone'])+0.1)
plt.plot(xx, np.median(bf_rec['n_electr'])-1*(xx-np.median(bf_rec['R_cone'])), '--', label='$n_e\propto R^{-1}$')
plt.plot(xx, np.median(bf_rec['n_electr'])-3/2*(xx-np.median(bf_rec['R_cone'])), ':', label='$n_e\propto R^{-3/2}$')
plt.legend()
plt.ylabel('Electron number density (cm$^{-3}$)')
plt.xlabel('Radius (cm)')
plt.pause(0.1)
plt.savefig('./plots/at2019dsg_Rn.pdf')
key = input('next?)')


# plt.clf()
# for i, mjd in enumerate(mjd_fit): 
# 	med, sig = np.log10(np.median(N_electron[i, :])), np.std(np.log10(N_electron[i,:]))
# 	plt.errorbar(mjd-mjd_fit[0]+12,med, sig,fmt='s')
# 	out_dict['N_electr'+str(int(mjd_fit[i]-mjd_fit[0]))] = np.array([med, sig])
# plt.ylabel('Electron number')
# plt.pause(0.1)
# key = input('next?)')


plt.clf()
plt.errorbar(bf_rec['time'], bf_rec['E_cone'], bf_rec['eE_cone'], fmt='o', label='Barniol, cone')
plt.errorbar(bf_rec['time'], bf_rec['E_SJ'], bf_rec['eE_SJ'], fmt='s', label='Sjoert, sphere')
plt.errorbar(bf_rec['time'], bf_rec['E_sphere'], bf_rec['eE_sphere'], fmt='x', label='Barniol, sphere')

xx = np.linspace(min(bf_rec['time'])-0.1, max(bf_rec['time'])+0.1)
plt.plot(xx, np.median(bf_rec['E_cone'])+np.log10(xx/np.median(bf_rec['time'])), '-', label='$E_{eq}\propto t$')
plt.ylabel('$E_{eq}$ (erg)')
plt.xlabel('time (day)')
plt.xscale('log')
plt.legend(loc=0)
xl = plt.xlim()
plt.legend(loc=2)
plt.pause(0.1)
plt.savefig('./plots/at2019dsg_tE.pdf')
key = input('next?)')

# plt.clf()
# for i, mjd in enumerate(mjd_fit): 
# 	med, sig = np.median(E_eq[i, :]), np.std(E_eq[i,:])
# 	plt.errorbar(mjd-mjd_fit[0], med, sig,fmt='s')
# 	out_dict['E_linear_'+str(int(mjd_fit[i]-mjd_fit[0]))] = np.array([med, sig])
# plt.ylabel('$E_{eq}$ (erg)')
# plt.pause(0.1)
# key = input('next?)')


plt.clf()
mid_time = bf_rec['time'][:-1]+(bf_rec['time'][1:]-bf_rec['time'][:-1])/2.
plt.errorbar(mid_time, bf_rec['v_cone'][:-1],bf_rec['ev_cone'][:-1], fmt='o',label='Barniol, cone')
plt.errorbar(mid_time, bf_rec['v_SJ'][:-1],bf_rec['ev_SJ'][:-1], fmt='s', label='Sjoert, sphere')
plt.errorbar(mid_time, bf_rec['v_sphere'][:-1],bf_rec['ev_sphere'][:-1], fmt='x', label='Barniol, sphere')
plt.legend(loc=0)
plt.ylabel('v/c')
plt.xlim(xl[0], xl[1])
plt.savefig('./plots/at2019dsg_tv.pdf')

# ---
# 
bb=np.mean(bf_rec['v_SJ'][0:2]) # bulk outflow for default model
bb = 0.5

# try different opening angle to find max outflow velocity
phi=45/180*pi # opening angle, free parameter
phi=60/180*pi # opening angle, free parameter
R0 = 10**np.mean(bf_rec['R_SJ'][0])  

fA_me = (1-cos(pi/4)) # about 0.3
fA=((1-cos(phi))/fA_me)

# scaling from Barniol Duran, Nakar, & Piran:
Rnew = R0 * (sjoert.stellar.beta2gamma(bb))**(10/17) / (fA)**(7/17.)  # note the Gamma^10/17 scaling doesn't matter when we are observing the jet edge on

print ('phi:', phi)
print ('fA:', fA)

# from Barniol Duran, Nakar, & Piran
print ('radius / v_bulk (day) on-axis:', Rnew*(1.05)*(1-bb)/(bb*3e10)/3600/24)

# here we assume we are observing the flux edge on
# and we correct for the increase/decrease of the inferred velocity with phi
bb *= Rnew/R0
print ('radius / v_bulk (day) off-axis:', Rnew*(1.05)/(bb*3e10)/3600/24)
print ('radius/t[0]/c                 :', Rnew*(1.05)/bf_rec['time'][0]/3600/24/3e10)  

# conclusion:
# for on-axis phi can get arbitrary small
# while of off-axis we have phi>0.1 to avoid v_bulk>c


