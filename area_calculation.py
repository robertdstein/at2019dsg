from scipy.integrate.quadrature import simps
from sjoert.stellar import radectolb
from numpy import pi

N=2000
ddec = np.linspace(90, -90, N)
rra =  np.linspace(0, 3600, N)
ll, bb  = radectolb(rra, ddec)
 
# apply survey footprint and plane cut
icuts = (np.abs(bb)>7)* (ddec>-30) * (ddec<85) 

# do integral
z = rra[:,None]*0 + np.sin(np.radians(90-ddec)) *icuts  
print ('square degree total:', simps(simps(z, np.radians(90-ddec)), np.radians(rra)) / (4*pi) *41252 )