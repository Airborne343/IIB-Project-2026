import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
from horsfield_model import l_m, d_m, Delta, A, h_m, c, compute_impedance


rho_g = 1.225 #kg/m^3

#Tree Structure
class Branch:
    def __init__(self, generation):
        self.generation = generation
        self.length = l_m[generation - 1]
        self.diameter = d_m[generation - 1]
        self.recursion = Delta[generation - 1]
        self.area = A[generation - 1]
        self.children = []
        
        self.flow = 0.0 #L/min
        self.pressure = 0.0 #Pa
        
        
    def representation(self):
        return f"Branch(n = {self.generation}, length = {self.length:.2f}, diameter = {self.diameter:.2f}, Δ = {self.recursion})"
    
def generate_children(parent_generation, lowgen_bifurcation_prob = 0.7):
    delta = Delta[parent_generation - 1]
    
    if parent_generation > 10:
        if delta == 0 or (parent_generation - delta) < 1:
            return []
        
        min_child_generation = max(1, (parent_generation - delta))
        max_child_generation = parent_generation - 1
        
        if max_child_generation < min_child_generation:
            return []
        
        elif max_child_generation == min_child_generation:
            return [min_child_generation, min_child_generation]
        
        child_1 = random.randint(min_child_generation, max_child_generation)
        child_2 = random.randint(min_child_generation, max_child_generation)
        
        return sorted([child_1, child_2])
    
    else:
        if parent_generation == 1:
            return []
        
        if random.random() < lowgen_bifurcation_prob:
            return [parent_generation - 1, parent_generation - 1]
        
        else:
            return []

def construct_tree(generation, max_recursion = 5, counter = 0):
    node = Branch(generation)
    
    if counter >= max_recursion or generation == 1:
        return node
    
    child_generations = generate_children(generation)
    
    if not child_generations:
        return node
    
    for gen in child_generations:
        if gen < generation:
            node.children.append(construct_tree(gen, counter + 1))
        else:
            node.children.append(construct_tree(gen, 0))
            
    return node

def distribute_flow(node, parent_flow):
    node.flow = parent_flow
    
    if not node.children:
        return
    
    areas = [child.area for child in node.children]
    total_area = sum(areas)
    
    for i, child in enumerate(node.children):
        child_flow = parent_flow * (areas[i]/total_area)
        distribute_flow(child, child_flow)

def get_terminal_paths_from(node, path = None, paths = None):
    if path is None:
        path = []
    if paths is None:
        paths = []
    
    path = path + [node]
    
    if not node.children: #if the node is a terminal branch
        paths.append(path)
        
    else:
        for child in node.children:
            get_terminal_paths_from(child, path, paths)
    return paths
            
def pressure_loss_across_one_tube(area, Q, Kc=0.05, Kd=0.35, area_ratio=0.2):
    Ac = area_ratio * area  # contraction area
    
    dp1 = 0.5 * rho_g * (Q**2) * ((1 + Kc)*(1/Ac**2) - (1/area**2))
    dp2 = rho_g * ((Q**2)/area) * ((1/area) - (Kd - 1)*(1/Ac))
    
    return (dp1 + dp2)

def compute_pressure_drop(path):
    total_pressure_drop = 0.0
    for i in range(len(path) - 1):
        flow = path[i].flow
        area = path[i].area
        delta_p = pressure_loss_across_one_tube(area, flow)
        total_pressure_drop += delta_p
    return total_pressure_drop

def print_tree(node, indent=0):
    print("  " * indent + f"n = {node.generation} | D = {node.diameter:.3f} cm | Q = {node.flow:.3e} L/min")
    for child in node.children:
        print_tree(child, indent + 1)

tree_root = construct_tree(35)
Q_range = np.linspace(0, 25, 26) #in L/min

max_pressure_drop_list = []
min_pressure_drop_list = []
mean_pressure_drop_list = []

Z_in = []

for q in Q_range:  
    print(f"Processing Q = {q:.2f} L/min")
    q = q/60000
    distribute_flow(tree_root, q)

    starting_nodes = tree_root.children
    all_pressure_drops = []

    for node in starting_nodes:
        paths = get_terminal_paths_from(node)
        for path in paths:
            drop = compute_pressure_drop(path)
            all_pressure_drops.append(drop)
    
    all_pressure_drops = [i/100 for i in all_pressure_drops]
    max_pressure_drop = max(all_pressure_drops)
    min_pressure_drop = min(all_pressure_drops)
    mean_pressure_drop = sum(all_pressure_drops)/len(all_pressure_drops)
    
    max_pressure_drop_list.append(max_pressure_drop)
    min_pressure_drop_list.append(min_pressure_drop)
    mean_pressure_drop_list.append(mean_pressure_drop)

pressure_loss_coeff_list = []

for i in range(len(Q_range) - 1):
    pressure_loss_coeff = mean_pressure_drop_list[i+1]/(0.5 * rho_g * ((Q_range[i+1] / (60000 * A[33])) ** 2))
    pressure_loss_coeff_list.append(pressure_loss_coeff)

plt.figure(figsize = (10,5))
plt.plot(Q_range, max_pressure_drop_list, label= "Maximum Pressure Drop")
plt.plot(Q_range, min_pressure_drop_list, label= "Minimum Pressure Drop")
plt.plot(Q_range, mean_pressure_drop_list, label= "Mean Pressure Drop")
plt.xlabel(r"Volumetric Flow Rate, Q ($\ell$/min)")
plt.ylabel(r"Total Pressure Loss, $\Delta$p (mbar)")
plt.title("Pressure-Flow Relationship")
plt.legend()
plt.grid(True)
plt.show()


for i, coeff in enumerate(pressure_loss_coeff_list):
    print(f"Q = {Q_range[i+1]:.2f} L/min → ζ = {coeff:.3e}")