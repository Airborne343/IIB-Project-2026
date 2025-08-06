import numpy as np
import matplotlib.pyplot as plt
from step01_horsfield_model import l_m, d_m, h_m, area, Delta, c, compute_impedance
from step02_tree_generation import get_depth_to_generation_map
from step04_inverse_laplace_transform import omega_sweep, impedance_n34

frequency = np.linspace(100, 2500, 2500)
omega_sweep = [freq * (2 * np.pi) for freq in frequency]
depth_to_generations = get_depth_to_generation_map()
generations_at_depth_2 = depth_to_generations.get(2, [])

def build_matrix(omega, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar):
    Z_real, Z_imag = compute_impedance(l_m, d_m, h_m, c, Delta, omega, N_T = 2350000, U_val = 0)
    Z_d = Z_real[33] + (1j * Z_imag[33])
    
    k = omega/c_gas
    ploss_factor = (zeta*u_bar/c_gas)
    impedance_factor = (Z_d/(rho*c_gas))
    
    matrix = np.zeros((8, 8), dtype = complex)
    
    #Equation 1
    matrix[0,0] = 1
    matrix[0,1] = 1
    
    #Equation 2
    matrix[1,0] = np.exp(1j * k * l0)
    matrix[1,1] = np.exp(-1j * k * l0)
    matrix[1,2] = -(1 + ploss_factor)
    matrix[1,3] = -(1 - ploss_factor)
    
    #Equation 3
    matrix[2,0] = S0 * np.exp(1j * k * l0)
    matrix[2,1] = -S0 * np.exp(-1j * k * l0)
    matrix[2,2] = -S1
    matrix[2,3] = S1
    
    #Equation 4
    matrix[3,2] = np.exp(1j * k * l1)
    matrix[3,3] = np.exp(-1j * k * l1)
    matrix[3,4] = -1
    matrix[3,5] = -1

    #Equation 5
    matrix[4,2] = np.exp(1j * k * l1)
    matrix[4,3] = np.exp(-1j * k * l1)
    matrix[4,6] = -1
    matrix[4,7] = -1
    
    #Equation 6
    matrix[5,2] = S1 * np.exp(1j * k * l1)
    matrix[5,3] = -S1 * np.exp(-1j * k * l1)
    matrix[5,4] = -S2
    matrix[5,5] = S2
    matrix[5,6] = -S3
    matrix[5,7] = S3

    #Equation 7
    matrix[6,4] = (1 - impedance_factor) * (np.exp(1j * k * l2))
    matrix[6,5] = (1 + impedance_factor) * (np.exp(-1j * k * l2))
    
    #Equation 8
    matrix[7,6] = (1 - impedance_factor) * (np.exp(1j * k * l3))
    matrix[7,7] = (1 + impedance_factor) * (np.exp(-1j * k * l3))
    return matrix

def determinant(omega, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar):
    matrix = build_matrix(omega, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar)
    return np.linalg.det(matrix)

def derivative(omega, delta, *args):
    f_plus = determinant(omega + delta, *args)
    f_minus = determinant(omega - delta, *args)
    return (f_plus - f_minus)/(2 * delta)

def Newton_Raphson(omega0, delta, tol, max_iter, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar):
    omega = omega0
    
    for i in range(max_iter):
        f = determinant(omega, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar)
        df = derivative(omega, delta, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar)
        
        print(f"[{i+1}/{max_iter}] ω = {omega:.6f}, |det| = {abs(f):.3e}, |Δω| = {abs(f/df):.3e}")

        if abs(df) == 0:
            raise ZeroDivisionError("Derivative is zero. Newton-Raphson step undefined.")
        
        omega_next = omega - f/df
        if abs(omega_next - omega) < tol:
            return omega_next
        
        omega = omega_next
    
    print("No convergence.")
    return None
    
l0 = l_m[34]/2
l1 = l_m[34]
l2 = l_m[(generations_at_depth_2[0] - 1)]
l3 = l_m[(generations_at_depth_2[-1] - 1)]
S0 = area[34]/2
S1 = area[34]
S2 = area[(generations_at_depth_2[0] - 1)]
S3 = area[(generations_at_depth_2[-1] - 1)]
rho = 1.225
c_gas = 343
zeta = 0.7
u_bar = 50

omega_roots = []
omega_guesses = np.arange(100, 2501, 10) * 2 * np.pi
delta = 1e-3
tol = 1e-6
max_iter = 100

for omega_init in omega_guesses:
    try:        
        omega_root = Newton_Raphson(
            omega_init, delta, tol, max_iter,
            l0 = l_m[34]/2, #parametrised in m
            l1 = l_m[34],
            l2 = l_m[(generations_at_depth_2[0] - 1)],
            l3 = l_m[(generations_at_depth_2[-1] - 1)],
            S0 = area[34]/2,   #parametrised in m^2
            S1 = area[34],
            S2 = area[(generations_at_depth_2[0] - 1)],
            S3 = area[(generations_at_depth_2[-1] - 1)],  
            rho = 1.225,
            c_gas = 343,
            zeta = 0.7,     #parametrised
            u_bar = 50      #parametrised
        )
             
        if omega_root is not None and omega_root > 1e-3:
            if not any(np.isclose(omega_root, r, rtol=0, atol=1) for r in omega_roots):
                omega_roots.append(omega_root)
                print(f"✅ Found eigenfrequency: {omega_root / (2 * np.pi):.2f} Hz")
        
    except ZeroDivisionError as e:
        print(f"Newton failed at {omega_init / (2 * np.pi):.1f} Hz — {e}")
        
    except Exception as e:
        print(f"Error at {omega_init / (2 * np.pi):.1f} Hz: {e}")

mode_shape_result = {}

print("Generations at depth 2:", generations_at_depth_2)
print("Unique eigenfrequencies (Hz):")
for i, omega_root in enumerate(sorted(omega_roots), start=1):
    frequency = omega_root/(2*np.pi)
    M = build_matrix(omega_root, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar)
    
    U, s, Vh = np.linalg.svd(M)
    mode_shape = Vh[-1, :]
    mode_shape /= np.max(np.abs(mode_shape))
    
    mode_shape_result[i] = {
        "frequency_Hz": frequency,
        "omega_root": omega_root,
        "determinant": determinant(omega_root, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar),
        "mode_shape": mode_shape
    }
    
    print(f"Mode {i}: {frequency :.2f} Hz")
    print(f"Determinant: {determinant(omega_root, l0, l1, l2, l3, S0, S1, S2, S3, rho, c_gas, zeta, u_bar)}")
    print(f"Mode Shape: {np.round(mode_shape, 3)}\n")