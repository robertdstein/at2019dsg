import numpy as np
from numpy import log10, log, exp

from sjoert.stellar import Planck, sigma_SB


nu_nuv = 3e8/2500e-10 
nu_g = 3e8/4770e-10

#nu_kc = 1479204846703550 # hardcoded to UVW2
nu_kc = nu_nuv

h_over_k =  6.6261e-27/1.3806e-16 

credit_int = 68.2689492137086

def gauss(x, sigma):
    
    out = np.exp( -0.5 * x**2 / (sigma**2) )
    return out

def cc_bol(T, nu):
    return Planck(nu, T)*nu /  (sigma_SB * T**4 / np.pi)


def cc_func(T, nu, nu_kc=nu_kc):
    
    to_kc = Planck(nu, T) /  Planck(nu_kc, T) * nu / nu_kc
    #to_kc =   (np.exp(h_over_k * nu_kc / T)-1) / (np.exp(h_over_k * nu / T)-1) * (nu/nu_kc)**4
    return to_kc

def gauss_exp(p, x, nu, model_name='nu_kc', *arg): 

    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # flux at peak
    b1 = 10**(p[2]) # gaussian rise
    
    leftside = a1*gauss(x-x_peak, b1)
    #leftside = a1*( (x-x_peak-b1)**(-2) ) * b1**2 # PL rise doesn't work

    # exponential decay
    a2 = a1 * gauss(0, b1)
    b2 = 10**(p[3]) # decay rate
    rightside = a2 * np.exp(-(x-x_peak)/b2) 
    
    leftside[x>x_peak]=0.
    rightside[x<=x_peak]=0. 
    
    both  = leftside+rightside

    # temperature + linear evolution
    T_pl = 10**p[4] #[10**p[4], p[5]]
    cc = cc_func(T_pl, nu)
    #T_pl = T_evo(x, p_T)

    return both * cc

def gauss_pl(p, x, nu, model_name='nu_kc', *arg): 

    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # flux at peak
    b1 = 10**(p[2]) # gaussian rise
    
    leftside = a1*gauss(x-x_peak, b1)
    #leftside = a1*( (x-x_peak-b1)**(-2) ) * b1**2 # PL rise doesn't work

    # exponential decay
    a2 = a1 * gauss(0, b1)
    b2 = 10**(p[3]) # decay rate
    rightside = a2 * np.exp(-(x-x_peak)/b2) 
    
    leftside[x>x_peak]=0.
    rightside[x<=x_peak]=0. 
    
    both  = leftside+rightside

    # temperature + linear evolution
    T_pl = 10**p[4] #[10**p[4], p[5]]
    cc = cc_func(T_pl, nu)
    #T_pl = T_evo(x, p_T)

    return both * cc


T_time = np.append(np.arange(-30, 400,30),np.arange(400, 1000,100))
L_time = np.append(np.arange(0, 400,30),np.arange(400, 1000,100))


def T_evo(x, p_T, T_time=T_time):
    
    
    if len(p_T)==2:
        
        T0, dT =p_T[0], p_T[1]
        #return np.clip(np.exp(np.log(T0) + dT * x ),1e2, 1e7)
        x_post = np.clip(x, 0, 1e99)
        return np.clip(T0 +  dT * x_post, 1e3, 1e5)
    
    else:

        return 10**np.interp(x, T_time, p_T)

def gauss_PL(p, x, nu, model_name='PL_bolo'): 

    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # luminosity at peak
    b1 = 10**(p[2]) # gaussian rise
    leftside = a1*gauss(x-x_peak, b1)

    # power-law
    a2 = a1 * gauss(0, b1)
    t0 = 10**p[3] # index
    alpha = p[4] # index
    rightside = a2 * ((x-x_peak+t0)/t0)**alpha
    
    leftside[x>x_peak]=0.
    rightside[x<=x_peak]=0. 
    
    both  = leftside+rightside

    
    # grid with fixed points 
    if model_name=='Tflex':
        p_T = p[5:]                 
    # temperature + linear evolution        
    else:
        p_T = [10**p[5], p[6]]  
   
    T_pl= T_evo(x, p_T) # keep t=0 fix to input for stablity

    if model_name=='PL':
        cc = cc_func(T_pl, nu) 
    elif (model_name=='Tflex') or (model_name=='PL_bolo'):
        cc = cc_bol(T_pl, nu) 
    else:
        print ('unknown model_name', model_name)

    return both * cc
