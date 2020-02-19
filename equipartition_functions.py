import sjoert
from sjoert import sync as sync_extra
from sjoert.stellar import beta2gamma, Doppler

import numpy as np
from numpy import pi, cos, sin, sqrt

from scipy.optimize import leastsq
import calendar
import scipy.interpolate
import scipy.stats


fit_all = True

# -----
# fit synchtroton model
# Eq. in Falcke99

i_obs = 60/180.*pi

gamma_max = 1e4
p_electron = 2.2

# some defaults
min_z = 100*3e5*1e6 # min inner jet radius 
phi = 1/10.0
#phi = 1/2.
Qindex = 1.0	# scaling between Xray lum and Qjet, used for inserting bumps
Lindex = 1.5 # how fast the jet power decreases with radius behing the jet head
Gindex = -1.0 # geometry for B~r scaling (-1 for perfect conical jet)
v_jet = 0.90
time0 = 60. # only used in move_jet
R0 = 1e15
sparse_fact = 0. # spacing between jet blobs

p_electron_single= 2.8
gamma_max_single = 1e4

this_delta = sjoert.stellar.Doppler(sjoert.stellar.beta2gamma(v_jet),i_obs=i_obs)
print ('Doppler factor:', this_delta)

z = 0.0512 # Bran...
D = sjoert.stellar.lumdis(z, h=0.7)

def sync99(nu0, B, r, D=D, p_electron=p_electron, Knorm=1.,
			gamma_max=gamma_max, gamma_min=1.0, 
			eps_e=1, filling_factor=1, delta=1):

	
	nu1 = nu0/delta

	kap1 = sync_extra.alpha_nu(B, nu1, p_electron, gamma_max, gamma_min, eps_e)
	eps1 = sync_extra.Ptot(B, nu1, p_electron, gamma_max, gamma_min, eps_e)
	return delta**2 * r**2 / (4*D**2) * eps1/kap1 * ( 1-np.exp(-kap1*r*filling_factor*Knorm) )	
	#return r**2 / (4*D**2) * eps1/kap1 * ( 1+np.exp(-kap1*r)*(r*kap1/6-1) )

def tau(nu0, B, r, p_electron=p_electron,
			gamma_max=gamma_max, gamma_min=1.0, Knorm=1,
			eps_e=1, filling_factor=1, delta=1):
	
	nu = nu0/delta
	
	if np.isscalar(B)==True:
		return r * sync_extra.alpha_nu(B, nu, p_electron, gamma_max, gamma_min, eps_e)*Knorm

	out = np.array(len(B))
	for i in range(len(B)):
		out[i] = r[i]*sync_extra.alpha_nu(B[i], nu[i], p_electron, gamma_max, gamma_min, eps_e)*Knorm[i]
	return out



def model_single(nu, B,r):
	#m = sync99(nu, B,r, p_electron=p_electron_single, gamma_max=gamma_max_single)
	m = model_sum(nu, B, r, v_jet=v_jet, p_electron=p_electron_single, gamma_max0=gamma_max_single, phi=phi, Lindex=1e99)
	return m

# sum of sperical sync blobs inside a conical geometry
nnr = 50
def model_sum(nu, B0, r0, Knorm0=1., gamma_max0=gamma_max, phi=phi, Qindex=Qindex, Gindex=Gindex, Lindex=Lindex, 
				v_jet=v_jet, i_obs=i_obs, p_electron=p_electron,sparse_fact=sparse_fact,
				return_array=False, bump=0):

	this_delta = sjoert.stellar.Doppler(sjoert.stellar.beta2gamma(v_jet),i_obs=i_obs)
	this_delta_counter = sjoert.stellar.Doppler(sjoert.stellar.beta2gamma(v_jet),i_obs=i_obs-pi)
	
	# put spheres in concial jet geometry
	z1 = r0 /phi 
	rt=np.zeros(nnr) ; rt[0] = r0
	
	#rr=np.zeros(nnr) ; rr[0] = z0*phi
	
	extra_space0 = v_jet*3e10 * 100 # 100 sec spacing at least
	extra_space = v_jet*3e10 * sparse_fact*24*3600 # option for gaps between the blobs
	
	for i in range(1,nnr):
	
		z1 -= rt[i-1] + rt[i-1]*(1-phi)/(1+phi)#+extra_space0+extra_space
		#rt[i] = (rt[i-1]*(1-phi)/(1+phi)-extra_space0-extra_space)
		rt[i] = z1*phi 

		#rr[i] = (z0-2*sum(rr[0:i])-extra_space0-extra_space)*phi*this_delta # multiply by Doppler factor because jet appears smaller in observer frame
		#print (rt[i], rr[i])
	#key = input()
	
	rr = rt*this_delta
	rr_counter = (rt-2*cos(i_obs)*rt*v_jet)*this_delta_counter  

	ircut =rr/phi>min_z
	ircut_counter =rr_counter/phi>min_z
	rr=rr[ircut]
	rr_counter=rr_counter[ircut_counter]
	if sum(ircut)==0:# savety
		rr = np.array([r0*phi*this_delta]) 
		rr_counter = np.array([r0*phi*this_delta_counter]) 
	if Lindex>9:		
	 	rr = np.array([r0]) # hack to single zone 
	 	rr_counter = np.array([r0]) # hack to single zone 


	B = B0 *(rr/rr[0])**(Gindex) 		# (modified) equipartition jet
	B *= sqrt( (rr/rr[0])**(Lindex) ) 	# deccreasing jet power
	Knorm1= Knorm0*(rr/rr[0])**((1-p_electron)*2/3.) 	# decrease of electron normalization due to cooling E_e~r**(-2/3), Knorm0 allows for scaling with respect to other epochs	

	
	B_counter = B0 *(rr_counter/rr[0])**(Gindex) 			# (modified) equipartition jet
	B_counter *= sqrt( (rr_counter/rr[0])**(Lindex) ) 		# deccreasing jet power
	Knorm1_counter= Knorm0*(rr_counter/rr[0])**((1-p_electron)*2/3.)

	Knorm1 = np.clip(Knorm1, 0, 1) # only cooling put to point where cooling starts 
	Knorm1_counter = np.clip(Knorm1, 0, 1)

	# gx1 = gamma_max0          +0*(rr/rr[0])**(-2/3.) 		# not used
	# gx1_counter = gamma_max0  +0*(rr_counter/rr[0])**(-2/3.) # not used


	# do bump of width 10 days
	if bump>0:
		t_r = (rr[0]-rr)/(v_jet*3e10)/24/3600/phi 
		Lincr = 1+0.1*sjoert.simstat.Gauss(t_r-bump, sigma=10)/sjoert.simstat.Gauss(0, sigma=10)
		B= B*sqrt(Lincr**Qindex)
		t_r = (rr_counter[0]-rr_counter)/(v_jet*3e10)/24/3600/phi 
		Lincr = 1+0.1*sjoert.simstat.Gauss(t_r-bump, sigma=10)/sjoert.simstat.Gauss(0, sigma=10)
		B_counter= B_counter*sqrt(Lincr**Qindex)

	m =np.zeros((len(rr),len(nu)))
	m_counter =np.zeros((len(rr_counter),len(nu)))	
	if return_array:
		tau_out =np.zeros((len(rr),len(nu)))
		tau_out_counter =np.zeros((len(rr_counter),len(nu)))

	for i in range(len(rr)):
		m[i,:] = sync99(nu, B[i], rr[i], 
							Knorm=Knorm1[i], gamma_max=gamma_max0, p_electron=p_electron, delta=this_delta) 
		if return_array:
			tau_out[i,:] = tau(nu, B[i], rr[i], delta=this_delta, gamma_max=gamma_max0, p_electron=p_electron)
	
	for i in range(len(rr_counter)):
		m_counter[i,:] = sync99(nu, B_counter[i], rr_counter[i],
							Knorm=Knorm1_counter[i], gamma_max=gamma_max0, p_electron=p_electron, delta=this_delta_counter)
		if return_array:
			tau_out_counter[i,:] = tau(nu, B[i], rr[i], delta=this_delta_counter, gamma_max=gamma_max0, p_electron=p_electron)

	if return_array:
		return rr, rr_counter, m,m_counter, tau_out, tau_out_counter
	
	return np.sum(m, axis=0) + np.sum(m_counter, axis=0)

# ---
# build a full jet model to fit all data simulatenously (is hard...)
beta2cmperday = 3e10*24*3600.

# function to move jet head forward, keeping track of B-field and cooling
def move_jet(time, B0, v_jet=v_jet, time0=time0, Gindex=Gindex, p_electron=p_electron, Lindex=Lindex, 
				i_obs=i_obs, gamma_max0=gamma_max):

	this_lorentz = sjoert.stellar.beta2gamma(v_jet)
	this_delta = sjoert.stellar.Doppler(this_lorentz, i_obs)

	R = v_jet*beta2cmperday *phi* (time-time0)*this_delta 	# jet head radius
	
	B = B0* (R/R0)**Gindex 								# Gindex==-1 for conical jet										
	Knorm0 = (R/R0)**((1-p_electron)*2/3.) 				# parametized to match Marscher Fpeak~nu_peak scaling
	gamma_maxR16 = gamma_max0 +0*(R/1e14)**(-2/3.) 			# not used
	if Lindex>9: # hack to force particle accelaration in hotspot model
		Knorm0 *=0+1


	return R, B, Knorm0, gamma_maxR16

def model_sumtime(nu, time, B0, time0=time0, Gindex=Gindex,gamma_max0=gamma_max, i_obs=i_obs,phi=phi,
					v_jet=v_jet, Lindex=Lindex, p_electron=p_electron, sparse_fact=sparse_fact, bump=0):


	R, B, Knorm0, gamma_maxR16 = move_jet(time, B0, time0=time0, v_jet=v_jet, Gindex=Gindex, Lindex=Lindex, i_obs=i_obs, gamma_max0=gamma_max0)
	
	out = np.zeros(len(time))
	for i in range(len(time)):
		out[i] = model_sum(np.array([nu[i]]), B[i], R[i], 
				Knorm0=Knorm0[i],gamma_max0=gamma_maxR16[i],
				phi=phi, Qindex=Qindex, Gindex=Gindex, Lindex=Lindex, i_obs=i_obs,
				v_jet=v_jet, p_electron=p_electron, sparse_fact=sparse_fact, bump=bump)

	return out

def res_single(p, nu,F, Ferr):	
	return (fit_single(nu, p)*1e23*1e3 - F)/Ferr
def fit_single(nu, p):
	if len(p)==3:
		return model_single(nu, p[0], p[1], p=p[2])
	
	return model_single(nu, p[0], p[1])
	

def res_sum(p, nu, K, gx, F, Ferr):
	return (fit_sum(nu,K, gx, p)*1e23*1e3 - F)/Ferr
def fit_sum(nu, K, gx, p):
	if len(p)==3:
		return model_sum(nu, p[0], p[1], Knorm0=K, gamma_max0=gx, p_electron=p[2])
	return model_sum(nu, p[0], p[1],Knorm0=K, gamma_max0=gx)

def fit_sumtime(nu, time, p, bump=0):

	_B0 =10**p[0]
	_time0=p[1]
	_v_jet=p[2]
	_Gindex=p[3]	
	_Lindex=p[4]	 
	_gamma_max=10**p[5]
	_phi = phi #10**p[3]
	_p_electron=p_electron
	_i_obs = i_obs#np.mod(p[4], pi)

	return model_sumtime(nu, time, _B0,time0=_time0, v_jet=_v_jet,Gindex=_Gindex,Lindex=_Lindex,
						p_electron=_p_electron, gamma_max0=_gamma_max, phi=_phi, i_obs=_i_obs, bump=bump)

def res_sumtime(p, nu, time, F, Ferr):
	
	this_res = (fit_sumtime(nu, time, p)*1e23*1e3- F)/Ferr
	
	for t in p:
		ss = '{0:0.4f} '.format(t)
		print (ss)
	ss = (' chi2/dof={0:0.4e}'.format(sum(this_res**2)/len(time)))
	print (ss)

	B0 = p[0]
	time0 = p[1]
	v_jet = p[2]
	G_index = p[3]
	Lindex = p[4]
	#i_obs = p[4]
	#phi = 10**p[3]

	# find peak of 16 GHz light curve
	ttest = np.linspace(30,150,50)
	lorentz_t = sjoert.stellar.beta2gamma(v_jet)
	delta_fit =sjoert.stellar.Doppler(lorentz_t, i_obs)
	delta_fit_counter =sjoert.stellar.Doppler(lorentz_t, i_obs-pi)
	model16 = fit_sumtime(np.repeat(16e9, len(ttest)), ttest, p)

	# find mean rad, weighted by optical depth
	Rt, Bt, Kt, _ = move_jet(130, B0, time0=time0, v_jet=v_jet, Gindex=Gindex) # get jet params at some observed time ~middel of AMI data
	rr,rr_c, msum,msum_c, mtau,mtau_c = model_sum(np.array([16e9]), Bt, Rt, Knorm0=Kt, Lindex=Lindex, return_array=True)
	rr = rr.flatten() ; mtau = mtau.flatten()
	ii = mtau<10
	tau_weighted_zdist = sum(rr[ii]*np.exp(-mtau[ii]))/sum(np.exp(-mtau[ii]))/phi

	ii = np.isnan(model16)==False
	if sum(ii):
		tmax = ttest[ii][np.argmax(model16[ii])]
		Z16 = v_jet*beta2cmperday * (tmax-time0)*delta_fit 	# z-coord of jet head when 16 GHz light curve peaks
		predicted_lag = tau_weighted_zdist/3e10*(1/v_jet/lorentz_t - cos(i_obs))/3600/24. 
		ss= '  time={0:0.1f}  z={1:0.1e} cm  lag={2:0.1f} d'.format(tmax, tau_weighted_zdist, predicted_lag)	
		print (ss)
		#this_res+= (predicted_lag-13)/5.*0.1#*len(ami0) # fit for a lag, or not
	else:
		print ('')

	print ('  delta_rat={0:0.1f}'.format(delta_fit**2 / delta_fit_counter**2))

	return this_res




