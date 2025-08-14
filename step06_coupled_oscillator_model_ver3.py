import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad

#__File_Imports__
from step01_horsfield_model import area, d_m, l_m
from step04_inverse_laplace_transform import all_poles, all_residues, d, state_space_model
from step05_mode_shapes import rho, c_gas, zeta, mode_shape_result

##notes:
#eta: modal amplitude
#u': acoustic velocity perturbation

#__Theoretical_Data__
flowspeed_data = pd.read_csv("_Volumetric Flow Rate - Default Dataset.csv")
flowspeed_t_data = flowspeed_data['t'].values
U_data = flowspeed_data['U'].values
dU_data = np.gradient(U_data, flowspeed_t_data)

volume_data = pd.read_csv("_Tidal Volume - Default Dataset.csv")
volume_t_data = volume_data['t'].values
Vh_data = volume_data['V_h'].values

U_interpolate = interp1d(flowspeed_t_data, U_data, kind='cubic', fill_value="extrapolate")
dU_interpolate = interp1d(flowspeed_t_data, dU_data, kind='cubic', fill_value="extrapolate")
Vh_interpolate = interp1d(volume_t_data, Vh_data, kind='cubic', fill_value="extrapolate")

def U(t): return U_interpolate(t)
def dU(t): return dU_interpolate(t)
def V_h(t): return Vh_interpolate(t)

results = {}
mode_data = mode_shape_result[1]
omega_0 = mode_data["omega_root"]
psi = mode_data["mode_shape"]
psi_j = psi[0]
frequency = np.linspace(100, 2500, 2500)
omega_sweep = [freq * (2 * np.pi) for freq in frequency]

_, _, _, A, B, C, D = state_space_model(all_poles, all_residues, d)

#__Alpha_Function__
def alpha_function(Z_d_val):
    trachea_area = area[33]
    psi_mag_squared = np.abs(psi_j) ** 2
    psi_norm_squared = np.trapz(np.abs(psi)**2, dx=1)

    if np.abs(Z_d_val) < 1e-10:
        print(f"Z_d too small or invalid at this step: {Z_d_val}")
        return 0.0
    
    alpha = -(rho * (c_gas**2) * psi_mag_squared * trachea_area)/(psi_norm_squared * Z_d_val)
    return alpha

def coupled_oscillatory_model(t, state):
    eta, eta_dot, u, u_dot = state[0], state[1], state[2], state[3]  #state space vector
    x = state[4 :]
    
    #parameters
    omega_d = c_gas * np.sqrt(area[33]/(V_h(t) * l_m[33]))   #damper frequency
    nu = 50                                                  #growth rate
    kappa = 1000                                             #non-linear coefficient
    beta = zeta/l_m[33]                                      #u_dot term in 2nd eqn
    gamma = - psi[0]/(rho * l_m[33])                         #eta_dot term in 2nd eqn
    
    x = np.array(x).reshape(-1, 1)
    u_input = np.array([[u]])
    x_dot = A @ x + B @ u_input
    x_dot = x_dot.flatten()
    
    #Z_t equation
    Z_t = np.real(C @ x + D * u_input)
    alpha = alpha_function(Z_t)
    
    #eta_ddot equation
    eta_ddot = - ((np.abs(omega_0) ** 2) * eta) - (alpha * eta_dot) + (2 * nu * eta_dot) - (kappa * (eta ** 2) * eta_dot)
    
    #u_ddot equation
    u_ddot = (-beta * (float(U(t)) + u) * u_dot - (abs(float(dU(t))) * beta + omega_d**2) * u - gamma * eta_dot)
    
    return [float(np.real(eta_dot).item()), float(np.real(eta_ddot).item()), float(np.real(u_dot).item()), float(np.real(u_ddot).item())] + np.real(x_dot).flatten().tolist()

#initialise
N = len(all_poles)
x_init = np.zeros(N)
x0 = [0.1, 0, 0.01, 0] + list(x_init)
t_span = (0, 1)
t_eval = np.linspace(*t_span, 51000)
solver = solve_ivp(coupled_oscillatory_model, t_span, x0, method = 'RK45', t_eval = t_eval)

results = {
    "omega_rad_s": omega_0,
    "frequency_Hz": omega_0 / (2 * np.pi),
    "eta(t)": solver.y[0],
    "u'(t)": solver.y[2],
    "t": solver.t,
    "max_eta": np.max(np.abs(solver.y[0])),
    "max_u": np.max(np.abs(solver.y[2])),
    "success": solver.success
}

plt.figure(figsize=(10, 5))
plt.plot(results["t"], results["eta(t)"], label="η(t)")
plt.plot(results["t"], results["u'(t)"], label="u'(t)")
plt.title("Coupled Response")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()