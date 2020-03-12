import numpy as np
from numpy import log10, log, exp

from sjoert.stellar import Planck, sigma_SB


nu_nuv = 3e8/2500e-10 
nu_g = 3e8/4770e-10

nu_kc = nu_g

h_over_k =  6.6261e-27/1.3806e-16 

credit_int = 68.2689492137086

def gauss(x, sigma):
    
    out = np.exp( -0.5 * x**2 / (sigma**2) )
    return out


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