import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

##notes:
#eta: modal amplitude
#u': acoustic velocity perturbation

data = pd.read_csv("Volumetric Flow Rate - Default Dataset.csv")
t_data = data['t'].values
U_data = data['U'].values
dU_data = np.gradient(U_data, t_data)

U_interpolate = interp1d(t_data, U_data, kind='cubic', fill_value="extrapolate")
dU_interpolate = interp1d(t_data, dU_data, kind='cubic', fill_value="extrapolate")

plt.figure(figsize = (10,5))
plt.plot(t_data[:67], U_data[:67], label= "U(t)")
plt.plot(t_data[:67], dU_data[:67], label= "dU(t)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity(m/s)")
plt.title("Base Flow")
plt.legend()
plt.grid(True)
plt.show()

def U(t):
    return U_interpolate(t)

def dU(t):
    return dU_interpolate(t)

def coupled_oscillatory_model(t, x):
    eta, eta_dot, u, u_dot = x  #state space vector 
    
    #parameters
    omega_0 = 200 * 2 * np.pi   #angular frequency
    omega_d = 50 * 2 * np.pi   #damper frequency
    nu = 50                     #growth rate
    kappa = 1000                #non-linear coefficient
    alpha = 500                 #u_dot term in 1st eqn
    beta = 7                    #u_dot term in 2nd eqn
    gamma = 100                 #eta_dot term in 2nd eqn
    
    #eta_ddot equation
    eta_ddot = - ((omega_0 ** 2) * eta) -(alpha * u_dot) + (2 * nu * eta_dot) - (kappa * (eta ** 2) * eta_dot)
    
    #u_ddot equation
    u_ddot = (-beta * U_interpolate(t) * u_dot - u * u_dot - (abs(dU_interpolate(t)) * beta + omega_d**2) * u - gamma * eta_dot)
    
    return [eta_dot, eta_ddot, u_dot, u_ddot]

#initialise
x0 = [0.1, 0, 0.01, 0]
t_span = (0, 100)
t_eval = np.linspace(*t_span, 100)

solver = solve_ivp(coupled_oscillatory_model, t_span, x0, method = 'RK45', t_eval = t_eval)

print("Solver success:", solver.success)
print("Solver message:", solver.message)
print("Max |η(t)|:", np.max(np.abs(solver.y[0])))
print("Max |u'(t)|:", np.max(np.abs(solver.y[2])))

plt.figure(figsize = (10,5))
plt.plot(solver.t, solver.y[0], label= "η(t)")
plt.plot(solver.t, solver.y[2], label= "u'(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (m)")
plt.title("Coupled Oscillatory Model")
plt.legend()
plt.grid(True)
plt.show()