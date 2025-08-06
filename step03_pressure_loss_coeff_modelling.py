import numpy as np
import scipy
import matplotlib.pyplot as plt
from step01_horsfield_model import area, l, Z_in, rho_g

Kc = 0.05 #assumed value from literature
Kd = 0.35 #value at Ac/A2 = 0.2
area_ratio = 0.2 #Ac/A2

def pressure_loss_across_one_tube(area, Q):
    #p1-p2 (in a single tube)
    Ac = area_ratio * area
    
    dp1 = 0.5 * rho_g * (Q**2) * ((1 + Kc)*(Ac**(-2)) - (area**(-2)))
    print(dp1)
    dp2 = rho_g * (Q**2/area) * ((1/area) - (Kd - 1)*(1/Ac))
    
    delta_p = dp1 + dp2
    
    return delta_p

pressure_loss_sweep = []
pressure_loss_coefficient = []

Q_range = np.linspace(0, 25, 26)
Q_range_SI = [i/60000 for i in Q_range]
Q_test = [10/60000]

for q in Q_test:
    total_pressure_loss = 0
    
    for i in range(len(area) - 1):
        delta_p = pressure_loss_across_one_tube(area[i], q)
        delta_p = delta_p/1000000
        total_pressure_loss += delta_p
    
    pressure_loss_sweep.append(total_pressure_loss)
    
    pressure_loss_coeff = total_pressure_loss/(0.5 * rho_g * ((q / area[33]) ** 2))
    pressure_loss_coefficient.append(pressure_loss_coeff)

plt.figure(figsize=(10,5))
plt.plot(Q_range, pressure_loss_coefficient)
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.xlabel(r"Volumetric Flow Rate, Q ($\ell$/min)")
plt.ylabel(r"Pressure Loss Coefficient")
plt.title("Pressure Loss Coeff - Volumetric Flow Relationship")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(Q_range, pressure_loss_sweep)
plt.xlabel(r"Volumetric Flow Rate, Q ($\ell$/min)")
plt.ylabel(r"Total Pressure Loss, $\Delta$p (MPa)")
plt.title("Pressure-Flow Relationship")
plt.grid(True)
plt.show()