import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##notes:
#eta: modal amplitude
#u': acoustic velocity perturbation



def U(t):
    return np.sin(2 * np.pi * t)

def dU(t):
    return 2 * np.pi * np.cos(2 * np.pi * t)

def coupled_oscillatory_model(t, x):
    eta, eta_dot, u, u_dot = x  #state space vector 
    
    #parameters
    omega_0 = 100 * 2 * np.pi   #angular frequency
    omega_d = 150 * 2 * np.pi   #damper frequency
    nu = 0.5                    #growth rate
    kappa = 0.1                 #non-linear coefficient
    alpha = 1.0                 #u_dot term in 1st eqn
    beta = 0.5                  #u_dot term in 2nd eqn
    gamma = 0.5                 #eta_dot term in 2nd eqn
    
    #eta_ddot equation
    eta_ddot = - ((omega_0 ** 2) * eta) -(alpha * u_dot) + (2 * nu * eta_dot) - (kappa * (eta ** 2) * eta_dot)
    
    #u_ddot equation
    u_ddot = (-beta * U(t) * u_dot - u * u_dot - (abs(dU(t)) * beta + omega_d**2) * u - gamma * eta_dot)
    
    return [eta_dot, eta_ddot, u_dot, u_ddot]

#initialise
x0 = [0.1, 0, 0.01, 0]
t_span = (0, 5)
t_eval = np.linspace(*t_span, 5000)

solver = solve_ivp(coupled_oscillatory_model, t_span, x0, method = 'RK45', t_eval = t_eval)

plt.figure(figsize = (10,5))
plt.plot(solver.t, solver.y[0], label= "η(t)")
plt.plot(solver.t, solver.y[2], label= "u'(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (m)")
plt.title("Coupled Oscillatory Model")
plt.legend()
plt.grid(True)
plt.show()