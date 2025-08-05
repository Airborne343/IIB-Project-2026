import numpy as np
import matplotlib.pyplot as plt
import skrf
from skrf.vectorFitting import VectorFitting
from scipy.signal import StateSpace, lsim
import sympy as sp
from step01_horsfield_model import l_m, d_m, h_m, Delta, c, compute_impedance

frequency = np.linspace(100, 2500, 2500)
omega_sweep = [freq * (2 * np.pi) for freq in frequency]

def impedance_n34(l, d, h, omega_range):
    Z_real34 = []
    Z_imag34 = []

    for omega in omega_range:
        Z_real, Z_imag = compute_impedance(l, d, h, c, Delta, omega, N_T = 2350000, U_val = 0)
        Z_real34.append(Z_real[33])
        Z_imag34.append(Z_imag[33])
        
    Z_complex = np.array(Z_real34)  + 1j * np.array(Z_imag34)
    
    return Z_complex

Z_cmplx = impedance_n34(l_m, d_m, h_m, omega_sweep)
Z_max = np.max(np.abs(Z_cmplx))
Z_scaled = Z_cmplx / Z_max

freq_Hz = np.array(omega_sweep)/(2*np.pi)
ntw = skrf.Network(f=freq_Hz, z=Z_scaled, z0=1)
s = 1j * np.array(omega_sweep)

real_range = range(2, 5)
complex_range = range(2, 9, 2)

best_error = np.inf
best_fit = None
best_params = None

for n_real in real_range:
    for n_complex in complex_range:
        try:
            vf = VectorFitting(ntw)
            vf.vector_fit(
                n_poles_real=n_real,
                n_poles_cmplx=n_complex,
                fit_constant=True,
                fit_proportional=True
            )

            Z_fit_scaled = np.zeros_like(s, dtype=complex)
            for p, r in zip(vf.poles, vf.residues[0]):
                Z_fit_scaled += r / (s - p)
            Z_fit = Z_max * Z_fit_scaled

            error = np.abs(Z_cmplx - Z_fit)
            mean_error = np.mean(error)

            print(f"[✓] real={n_real}, complex={n_complex} → mean abs error: {mean_error:.4e}")

            if mean_error < best_error:
                best_error = mean_error
                best_fit = Z_fit
                best_params = (n_real, n_complex)

        except Exception as e:
            print(f"[x] real={n_real}, complex={n_complex} → FAILED: {e}")

print(f"Constant coefficient: {vf.constant_coeff}")
print(f"Z_max: {Z_max}")

print(f"\n✅ Best Fit: real={best_params[0]}, cmplx={best_params[1]} → mean error: {best_error:.4e}")

print("\n🔍 Transfer Function (Best Fit):")
vf_best = VectorFitting(ntw)
vf_best.vector_fit(
    n_poles_real=best_params[0],
    n_poles_cmplx=best_params[1],
    fit_constant=True,
    fit_proportional=True
)

s, t = sp.symbols('s t')
poles = vf_best.poles
residues = vf_best.residues[0]
d = vf_best.constant_coeff[0]
e = vf_best.proportional_coeff[0]

def include_conjugates(poles, residues):
    all_poles = []
    all_residues = []
    
    for p, r in zip(poles, residues):
        all_poles.append(p)
        all_residues.append(r)
        
        if np.imag(p) > 0:
            all_poles.append(np.conj(p))
            all_residues.append(np.conj(r))
    
    return np.array(all_poles), np.array(all_residues)

all_poles, all_residues = include_conjugates(poles, residues)
    
def state_space_model(poles, residues, d, t = None):
    #_____State-Space Model_____#
    A = np.diag(poles)
    B = np.ones((len(poles), 1))
    C = residues.reshape(1, -1)
    D = 0

    if t is None:
        t = np.linspace(0, 0.2, 2000)

    dt = t[1] - t[0]
    u = np.zeros_like(t)
    u[0] = 1 / dt
        
    sys = StateSpace(A, B, C, D)
    t_out, y_out, x_out = lsim(sys, U=u, T=t)
    
    return t_out, y_out, x_out

t_impulse, Z_t, x_out = state_space_model(all_poles, all_residues, d)

plt.figure()
plt.plot(t_impulse, Z_t.real, label="Re[Z(t)]")
plt.title("Impulse Response Z(t)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.show()
    
plt.figure(figsize=(12, 5))
plt.plot(omega_sweep, Z_cmplx.real, label="Original Z_real")
plt.plot(omega_sweep, Z_cmplx.imag, label="Original Z_imag")
plt.plot(omega_sweep, best_fit.real, '--', label="Fitted Z_real")
plt.plot(omega_sweep, best_fit.imag, '--', label="Fitted Z_imag")
plt.xlabel("ω (rad/s)")
plt.ylabel("Impedance")
plt.title(f"Best Fit (real={best_params[0]}, cmplx={best_params[1]})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()