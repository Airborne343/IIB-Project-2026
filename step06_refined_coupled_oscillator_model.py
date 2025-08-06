import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

#__File_Imports__
from step01_horsfield_model import area
from step04_inverse_laplace_transform import all_poles, all_residues, d, state_space_model
from step05_mode_shapes import rho, c_gas, zeta, l1, mode_shape_result

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

#__Initialise__
x0 = [0.1, 0, 0.01, 0]
t_span = (0, 0.2)
t_eval = np.linspace(*t_span, 5000)

#__Mode_Shape_&_Frequency_Data
mode_data = mode_shape_result[0]
omega0 = mode_data["omega_root"]
psi = mode_data["mode_shape"]
psi_j = psi[0]
frequency = np.linspace(100, 2500, 2500)
omega_sweep = [freq * (2 * np.pi) for freq in frequency]

#__Parameters__
nu = 50                                                                 #growth rate
kappa = 1000                                                            #non-linear coefficient
beta = zeta/l1                                                          #u_dot term in 2nd eqn
gamma = -psi_j/(rho * l1)                                               #eta_dot term in 2nd eqn

#__Storage_Lists__
eta_array = np.zeros(len(t_eval) + 1)
eta_dot_array = np.zeros(len(t_eval) + 1)
u_array = np.zeros(len(t_eval) + 1)
u_dot_array = np.zeros(len(t_eval) + 1)

eta_array[0] = x0[0]
eta_dot_array[0] = x0[1]
u_array[0] = x0[2]
u_dot_array[0] = x0[3]

#__Alpha_Function__
def alpha_function(Z_d_val, Vh_val):
    trachea_area = area[33]
    psi_mag_squared = np.abs(psi_j) ** 2
    psi_norm_squared = np.trapz(np.abs(psi)**2, dx=1)
    alpha = -(rho * (c_gas**2) * psi_mag_squared * trachea_area)/(psi_norm_squared * Z_d_val)
    return alpha
    
#_Coupled_Oscillator_Loop
x_curr = np.zeros((len(all_poles),)) #State-Space initial condition

for n in range(len(t_eval) - 1):
    t0, t1 = t_eval[n], t_eval[n + 1]
    dt = t1 - t0
    
    u_curr, u_dot_curr = u_array[n], u_dot_array[n]
    eta_curr, eta_dot_curr = eta_array[n], eta_dot_array[n]
    omega_d = c_gas * np.sqrt(area[33]/(V_h(t0) * l1))
    
    _, Z_t, _, A, B, C, D = state_space_model(all_poles, all_residues, d, t_eval, u_curr)
    Z_d_val = Z_t[n] #Z_d(t) at time t0
    alpha_val = alpha_function(Z_d_val, V_h(t0))
    
    def eta_dot_guess(t): return eta_dot_curr
    def u_dot_guess(t): return u_dot_curr
    
    for iteration in range(10):
        #__Solve_2nd_coupled_oscillator_equation__
        def u_system(t, y):
            u, u_dot = y
            eta_dot_val = eta_dot_guess(t)
            u_ddot = (-beta * U(t) * u_dot - u * u_dot - (abs(dU(t)) * beta + omega_d**2) * u - gamma * eta_dot_val(t))
            
            return [u_dot, u_ddot]
        
        sol_u = solve_ivp(u_system, [t0, t1], [u_curr, u_dot_curr], t_eval=[t1])
        u_new, u_dot_new = sol_u.y[:, -1]
        
        #__Update_State_Space_Model__
        x_dot = A @ x_curr + B.flatten() * u_new
        x_new = x_curr + (x_dot * dt)
        Z_d = (C @ x_new + D * u_new).item()
        
        #__Update_alpha_value__
        alpha_new = alpha_function(Z_d, V_h(t0))
        
        #__Solve_1st_coupled_oscillator_equation__
        def eta_system(t, y):
            eta, eta_dot = y
            eta_ddot = - ((omega0 ** 2) * eta) - (alpha_new * u_dot_new) + (2 * nu * eta_dot) 
            - (kappa * (eta ** 2) * eta_dot)
            return [eta_dot, eta_ddot]
        
        sol_eta = solve_ivp(eta_system, [t0, t1], [eta_curr, eta_dot_curr], t_eval=[t1])
        eta_new, eta_dot_new = sol_eta.y[:, -1]
        
        #__Update_values_for_next_iteration__
        eta_dot_guess = lambda t, val=eta_dot_new: val
        u_curr, u_dot_curr = u_new, u_dot_new
        eta_curr, eta_dot_curr = eta_new, eta_dot_new
        x_curr = x_new

    #__Store_final_values__
    u_array[n + 1], u_dot_array[n + 1] = u_new, u_dot_new
    eta_array[n + 1], eta_dot_array[n + 1] = eta_new, eta_dot_new
    
plt.plot(t_eval, eta_array[:-1], label="η(t)")
plt.plot(t_eval, u_array[:-1], label="u(t)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Coupled Acoustic-Oscillator Response")
plt.grid(True)
plt.show()