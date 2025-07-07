#the point of this code is to relate the impedance of the horsfield model for the lumped model we are using the coupled oscillator model to the pressure losses in the trachea

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.special

Z_in = []
l = [0.0600, 0.0600, 0.0600, 0.0600, 0.0600, 
     0.0737, 0.0938, 0.1313, 0.1638, 0.1375, 
     0.3125, 0.3875, 0.4500, 0.5250, 0.6000, 
     0.6462, 0.7875, 0.8000, 0.9625, 1.0125, 
     1.0250, 1.1500, 1.0000, 1.2375, 1.1875, 
     1.0750, 1.3500, 1.2125, 1.4125, 1.4125, 
     1.3125, 1.3750, 2.7500, 6.2500, 12.500] #in cm

d = [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 
     0.1000, 0.0537, 0.0600, 0.0663, 0.0788,
     0.0950, 0.1189, 0.1375, 0.1750, 0.2000,
     0.2250, 0.2500, 0.2725, 0.3000, 0.3125,
     0.3375, 0.2500, 0.3625, 0.3875, 0.4375,
     0.4375, 0.5375, 0.6750, 0.7375, 0.7375,
     0.9125, 1.0000, 1.3750, 1.5000, 2.0000] #in cm

A = []
for n in range(len(d)):
    area = np.pi/4 * (d[n] ** 2)
    A.append(area)
    
a = []
for n in range(len(d)):
    radius = d/2
    a.append(radius)

#Constant Parameters (set to 37.8C)
rho_g = 1.225 #kg/m^3
c_g = 340 #m/s
eta_g = 1.9 * (10**(-5)) #kg/(m*s)
kappa_g = 0.02735 #W/mK
C_g = 1005 #kJ/kg K

#Variable Parameters
omega = 1000

#initialise lists
Z_0 = []
gamma_0 = []
Z = []
Y = []
F_v = []
F_t = []
Z_T = []
Z_w = []

#Characteristic Equations
for n in range(1,35):
    F_t[n] = 2/(a[n] * np.sqrt((-1j * omega * C_g/kappa_g))) * ((scipy.special.jv(1, (a[n] * np.sqrt((-1j * omega * C_g/kappa_g)))))/(scipy.special.jv(0, (a[n] * np.sqrt((-1j * omega * C_g/kappa_g)))))) #Equation 7
    F_v[n] = 2/(a[n] * np.sqrt((-1j * omega * rho_g/eta_g))) * ((scipy.special.jv(1, (a[n] * np.sqrt((-1j * omega * rho_g/eta_g)))))/(scipy.special.jv(0, (a[n] * np.sqrt((-1j * omega * rho_g/eta_g)))))) #Equation 6
    
    Y[n] = ((1j * omega * A[n])/(rho_g * (c_g **2))) * (1 + (0.402 * F_t[n])) + ((Z_w[n] * l[n])**(-1)) #Equation 5
    Z[n] = (1j * omega * rho_g)/(A[n] * (1 - F_v[n]))                                                   #Equation 4
    
    
    gamma_0[n] = np.sqrt(Z[n] * Y[n])   #Equation 3
    Z_0[n] = np.sqrt(Z[n]/Y[n])         #Equation 2
    
    Z_in[n] = (Z_T[n] + Z_0[n] * np.tanh(gamma_0[n] * l[n]))/(1 + (Z_T[n]/Z_0[n]) * np.tanh(gamma_0[n] * l[n])) #Equation 1