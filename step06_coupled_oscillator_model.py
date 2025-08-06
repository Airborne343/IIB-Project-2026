import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from step05_mode_shapes import rho, c_gas, zeta, l1, mode_shape_result
from step01_horsfield_model import area

##notes:
#eta: modal amplitude
#u': acoustic velocity perturbation

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

plt.figure(figsize = (10,5))
plt.plot(flowspeed_t_data[:67], U_data[:67], label= "U(t)")
plt.plot(flowspeed_t_data[:67], dU_data[:67], label= "dU(t)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Base Flow")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize = (10,5))
plt.plot(volume_t_data[:48], Vh_data[:48], label= "Vh(t)")
plt.xlabel("Time (s)")
plt.ylabel("Volume (m^3)")
plt.title("Tidal Volume")
plt.legend()
plt.grid(True)
plt.show()

def U(t):
    return U_interpolate(t)

def dU(t):
    return dU_interpolate(t)

def V_h(t):
    return Vh_interpolate(t)

results = {}

for i, mode_data in mode_shape_result.items():
    omega_0 = mode_data["omega_root"]
    psi = mode_data["mode_shape"]
    
    def coupled_oscillatory_model(t, x):
        eta, eta_dot, u, u_dot = x  #state space vector 
        
        #parameters
        omega_d = c_gas * np.sqrt(area[33]/(V_h(t) * l1))      #damper frequency
        nu = 50                                             #growth rate
        kappa = 1000                                        #non-linear coefficient
        alpha = 500                                         #u_dot term in 1st eqn
        beta = zeta/l1                                      #u_dot term in 2nd eqn
        gamma = -psi[0]/(rho * l1)                          #eta_dot term in 2nd eqn
        
        #eta_ddot equation
        eta_ddot = - ((omega_0 ** 2) * eta) -(alpha * u_dot) + (2 * nu * eta_dot) - (kappa * (eta ** 2) * eta_dot)
        
        #u_ddot equation
        u_ddot = (-beta * U(t) * u_dot - u * u_dot - (abs(dU(t)) * beta + omega_d**2) * u - gamma * eta_dot)
        
        return [eta_dot, eta_ddot, u_dot, u_ddot]

    #initialise
    x0 = [0.1, 0, 0.01, 0]
    t_span = (0, 0.2)
    t_eval = np.linspace(*t_span, 5000)

    solver = solve_ivp(coupled_oscillatory_model, t_span, x0, method = 'RK45', t_eval = t_eval)

    results[i] = {
        "omega_rad_s": omega_0,
        "frequency_Hz": omega_0 / (2 * np.pi),
        "eta(t)": solver.y[0],
        "u'(t)": solver.y[2],
        "t": solver.t,
        "max_eta": np.max(np.abs(solver.y[0])),
        "max_u": np.max(np.abs(solver.y[2])),
        "success": solver.success
    }
    print(f"✅ Mode {i} | f = {omega_0 / (2 * np.pi):.2f} Hz | Max |η| = {results[i]['max_eta']:.3e}")

for i, data in results.items():
    plt.figure(figsize=(10, 5))
    plt.plot(data["t"], data["eta(t)"], label="η(t)")
    plt.plot(data["t"], data["u'(t)"], label="u'(t)")
    plt.title(f"Mode {i} — f = {data['frequency_Hz']:.2f} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()