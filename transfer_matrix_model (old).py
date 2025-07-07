import numpy as np
import matplotlib.pyplot as plt

#needed constants
rho = 0
area = 0
c = 0
freq = 0
omega = 2 * np.pi * freq

Z0 = rho * c/area      #characteristic impedance of uniform acoustic duct

def duct_transfer_matrix(length, k):
    T_duct = np.array([
        [np.exp(1j*k*length), 0],
        [0, np.exp(-1j*k*length)]
    ])
    
    return T_duct

def area_change_transfer_matrix(alpha1):
    T_area = 0.5 * np.array([
        [1 + alpha1, 1 - alpha1],
        [1 - alpha1, 1 + alpha1]    
    ])
    
    return T_area

def pressure_loss_transfer_matrix(alpha1, zeta, M):
    T_ploss = 0.5 * np.array([
        [1 + alpha1 - zeta*M, 1 - alpha1 + zeta*M],
        [1 - alpha1 + zeta*M, 1 + alpha1 - zeta*M]    
    ])
    
    return T_ploss

T_sum = np.eye(2)
T_list = [T1, T2,...TN] #list of transfer matrices
for T in T_list:
    T_sum = np.dot(T, T_sum)
    
W = np.array([f, g])
