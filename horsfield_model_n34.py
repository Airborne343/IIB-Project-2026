import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from horsfield_model import l_m, d_m, h_m, Delta, c, A, a, compute_impedance
import skrf

Z_real34 = []
Z_imag34 = []

frequency = np.linspace(100, 2500, 2500)
omega_sweep = [freq * (2 * np.pi) for freq in frequency]

for omega in omega_sweep:
    Z_real, Z_imag = compute_impedance(l_m, d_m, h_m, c, Delta, omega, N_T = 2350000, U_val = 0)
    Z_real34.append(Z_real[33])
    Z_imag34.append(Z_imag[33])

plt.figure(figsize = (12,5))
plt.plot(omega_sweep, Z_real34, label = "Z_real")
plt.plot(omega_sweep, Z_imag34, label = "Z_imag")
plt.xlabel("omega (ω)")
plt.ylabel("Impedance")
plt.title("Impedance at n = 34 generation")
plt.legend()
plt.grid(True)
plt.show()