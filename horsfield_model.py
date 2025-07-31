#the point of this code is to relate the impedance of the horsfield model for the lumped model we are using the 
#coupled oscillator model to the pressure losses in the trachea

#primary research paper link: https://doi.org/10.1016/j.jsv.2014.11.026

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.special

#length (in cm)
l_cm = [0.0600, 0.0600, 0.0600, 0.0600, 0.0600, 
     0.0737, 0.0938, 0.1313, 0.1638, 0.1375, 
     0.3125, 0.3875, 0.4500, 0.5250, 0.6000, 
     0.6462, 0.7875, 0.8000, 0.9625, 1.0125, 
     1.0250, 1.1500, 1.0000, 1.2375, 1.1875, 
     1.0750, 1.3500, 1.2125, 1.4125, 1.4125, 
     1.3125, 1.3750, 2.7500, 6.2500, 12.500]

#diameter (in cm)
d_cm = [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 
     0.1000, 0.0537, 0.0600, 0.0663, 0.0788,
     0.0950, 0.1189, 0.1375, 0.1750, 0.2000,
     0.2250, 0.2500, 0.2725, 0.3000, 0.3125,
     0.3375, 0.2500, 0.3625, 0.3875, 0.4375,
     0.4375, 0.5375, 0.6750, 0.7375, 0.7375,
     0.9125, 1.0000, 1.3750, 1.5000, 2.0000]

#recursion value
Delta = [0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         1, 2, 2, 3, 3,
         3, 3, 3, 3, 3,
         3, 3, 3, 3, 3,
         3, 3, 3, 3, 3,
         3, 3, 3, 2, 1]

#wall thickness (in cm) - Note: this value, along with l and d, differ from the reference paper used for c.
h_cm = [0.0065, 0.0065, 0.0065, 0.0065, 0.0065,
     0.0065, 0.0036, 0.0040, 0.0045, 0.0052,
     0.0063, 0.0075, 0.0084, 0.0061, 0.0106,
     0.0114, 0.0120, 0.0125, 0.0131, 0.0134,
     0.0139, 0.0140, 0.0143, 0.0147, 0.0158,
     0.0158, 0.0186, 0.0256, 0.0305, 0.0305,
     0.0511, 0.0660, 0.1685, 0.2169, 0.4655]

#fractional proportions of cartilage (taken from reference 21 in the primary research paper used above)
c = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0224, 0.0262, 0.0309, 0.0329,
     0.0370, 0.0390, 0.0410, 0.0450, 0.0526,
     0.0526, 0.0671, 0.0851, 0.0926, 0.2000,
     0.2500, 0.3300, 0.5000, 0.5000, 0.6700]

#convert all above parameters to m
l_m = [i * 0.01 for i in l_cm]
d_m = [i * 0.01 for i in d_cm]
h_m = [i * 0.01 for i in h_cm]

#cross-sectional area (in m^2)
A = []
for n in range(len(d_m)):
    area = np.pi/4 * (d_m[n] ** 2)
    A.append(area)

#radius (in m)
a = []
for n in range(len(d_m)):
    radius = d_m[n]/2
    a.append(radius)

def compute_impedance(l, d, h, c, Delta, omega = 1000, N_T = 2350000, U_val = None):
    #Constant Parameters (set to 37.8C)
    rho_g = 1.225 #kg/m^3 - air density
    c_g = 343 #m/s - speed of sound
    eta_g = 1.82 * (10**(-5)) #kg/(m*s) - air viscosity
    kappa_g = 6.4 * (10**(-3)) #W/mK - air thermal conductivity
    C_g = 240 #kJ/kg K - air specific heat capacity

    #double check these values
    rho_soft_tissue = 1060 #kg/m^3
    rho_cartilage = 1140 #kg/m^3

    cartilage_viscosity = 688.0     #Pa s
    soft_tissue_viscosity = 102.0   #Pa s

    E_cartilage = 3.92 * (10**5)    #Pa
    E_soft_tissue = 5.81 * (10**4)  #Pa

    Rt = 372.5              #Pa s m^-3
    It = 55.9               #Pa s^2 m^-3
    Ct = 1.275 * (10**(-5)) #m^3 Pa^-1

    #Variable Parameters
    k = omega/c_g

    #initialise lists
    Z_in = []       #inlet impedance
    Z_0 = []        #characteristic impedance of airway segment
    gamma_0 = []    #propagation coefficient of airway segment
    Z = []          #series impedance of airway segment
    Y = []          #shunt admittance of airway segment
    F_v1 = []       #sound attenuation due to air viscosity
    F_v2 = []
    F_t1 = []       #sound attenuation due to thermal dissipation
    F_t2 = []
    Z_T = []        #acoustic impedance at the far end of each segment
    Z_w = []        #effective volumetric wall impedance
    Z_wc = []       #effective volumetric wall impedance (cartilage)
    Z_ws = []       #effective volumetric wall impedance (soft-tissue)

    R_wc = []       #series resistance (cartilage)
    R_ws = []       #series resistance (soft-tissue)
    I_wc = []       #inertance (cartilage)
    I_ws = []       #inertance (soft-tissue)
    C_wc = []       #compliance (cartilage)
    C_ws = []       #compliance (soft-tissue)

    #Characteristic Equations
    for n in range(len(d)):
        if n == 0:
            ZT = N_T/((1j * omega * C_g)+(1/(Rt + 1j*((omega * It)-1/(omega * Ct)))))
            Z_T.append(ZT)
        
        else:
            ZT = 1/((1/Z_T[n-1]) + (1/Z_T[n-1-Delta[n]]))
            Z_T.append(ZT)
        
        Rwc = (4 * h[n] * cartilage_viscosity)/(np.pi * (d[n]**3) * l[n])   #Equation 10a
        Rws = (4 * h[n] * soft_tissue_viscosity)/(np.pi * (d[n]**3) * l[n]) #Equation 10b
        Iwc = (h[n] * rho_cartilage)/(np.pi * d[n] * l[n])                  #Equation 11a
        Iws = (h[n] * rho_soft_tissue)/(np.pi * d[n] * l[n])                #Equation 11b
        Cwc = (np.pi * (d[n]**3) * l[n])/(4 * h[n] * E_cartilage)           #Equation 12a
        Cws = (np.pi * (d[n]**3) * l[n])/(4 * h[n] * E_soft_tissue)         #Equation 12b
        R_wc.append(Rwc)
        R_ws.append(Rws)
        I_wc.append(Iwc)
        I_ws.append(Iws)
        C_wc.append(Cwc)
        C_ws.append(Cws)
        
        Zwc = R_wc[n] + 1j * ((omega * I_wc[n])- (1/(omega * C_wc[n]))) #Equation 9a
        Zws = R_ws[n] + 1j * ((omega * I_ws[n])- (1/(omega * C_ws[n]))) #Equation 9b
        Z_wc.append(Zwc)
        Z_ws.append(Zws)
        
        Zw = ((c[n]/Z_wc[n]) + ((1-c[n])/Z_ws[n]))**(-1)    #Equation 8
        Z_w.append(Zw)
        
        Ft1 = 2/(a[n] * np.sqrt((-1j * (omega - (k*U_val)) * C_g/kappa_g))) * ((scipy.special.jv(1, (a[n] * np.sqrt((-1j * (omega - (k*U_val)) * C_g/kappa_g)))))/(scipy.special.jv(0, (a[n] * np.sqrt((-1j * (omega - (k*U_val)) * C_g/kappa_g)))))) #Equation 7
        Ft2 = 2/(a[n] * np.sqrt((-1j * (omega + (k*U_val)) * C_g/kappa_g))) * ((scipy.special.jv(1, (a[n] * np.sqrt((-1j * (omega + (k*U_val)) * C_g/kappa_g)))))/(scipy.special.jv(0, (a[n] * np.sqrt((-1j * (omega + (k*U_val)) * C_g/kappa_g)))))) #Equation 7
        Fv1 = 2/(a[n] * np.sqrt((-1j * (omega - (k*U_val)) * rho_g/eta_g))) * ((scipy.special.jv(1, (a[n] * np.sqrt((-1j * (omega - (k*U_val)) * rho_g/eta_g)))))/(scipy.special.jv(0, (a[n] * np.sqrt((-1j * (omega - (k*U_val)) * rho_g/eta_g)))))) #Equation 6
        Fv2 = 2/(a[n] * np.sqrt((-1j * (omega + (k*U_val)) * rho_g/eta_g))) * ((scipy.special.jv(1, (a[n] * np.sqrt((-1j * (omega + (k*U_val)) * rho_g/eta_g)))))/(scipy.special.jv(0, (a[n] * np.sqrt((-1j * (omega + (k*U_val)) * rho_g/eta_g)))))) #Equation 6
        F_t1.append(Ft1)
        F_t2.append(Ft2)
        F_v1.append(Fv1)
        F_v2.append(Fv2)
        
        y = (((1j * (omega - (k*U_val)) * A[n])/(rho_g * (c_g **2))) * (1 + (0.402 * F_t1[n])) + ((Z_w[n] * l[n])**(-1))) + ((1j * (omega + (k*U_val)) * A[n])/(rho_g * (c_g **2))) * (1 + (0.402 * F_t2[n])) + ((Z_w[n] * l[n])**(-1))               #Equation 5
        z = (1j * (omega - (k*U_val)) * rho_g)/(A[n] * (1 - F_v1[n])) + (1j * (omega + (k*U_val)) * rho_g)/(A[n] * (1 - F_v2[n]))                                                                                                                     #Equation 4
        Y.append(y)
        Z.append(z)
        
        gamma0 = np.sqrt(Z[n] * Y[n])   #Equation 3 - propagation coefficient
        Z0 = np.sqrt(Z[n]/Y[n])         #Equation 2 - characteristic impedance
        gamma_0.append(gamma0)
        Z_0.append(Z0)
        
        Zin = (Z_T[n] + Z_0[n] * np.tanh(gamma_0[n] * l[n]))/(1 + (Z_T[n]/Z_0[n]) * np.tanh(gamma_0[n] * l[n])) #Equation 1
        Z_in.append(Zin)
        
    Z_real = np.real(Z_in)
    Z_imag = np.imag(Z_in)
        
    return Z_real, Z_imag

Z_real, Z_imag = compute_impedance(l_m, d_m, h_m, c, Delta, omega = 1000, N_T = 2350000, U_val = 100)

# understanding impedance:
# Z = R + iX
# R: resistance -> any energy loss (viscous, thermal etc.). changes how oscillations grow or decay
# X: impedance -> changes frequency of oscillations