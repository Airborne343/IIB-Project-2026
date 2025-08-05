import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import random
from vedo import Cylinder, Plotter, Sphere

#01_Scaling
def bronchi_scaling(order, D1 = 0.007, L1 = 0.1, ratio_d = 1.7, ratio_l = 1.3):
    #computing the scaled diameter and length of every generation bronchi and alveolus
    D = D1 * (ratio_d ** (order-1))
    L = L1 * (ratio_l ** (order-1))
    A = np.pi * ((D/2) **2)
    return D, L, A

#02_Geometric_Modelling
def tree_generator(max_order = 17):
    #building a trachea, bronchi, alveolus tree
    tree = {}
    node_counter = [0]
    
    def node_generator(order, parent_id = None):
        node_id = node_counter[0]
        node_counter[0] += 1
        
        D, L, A = bronchi_scaling(order)
        tree[node_id] = {
            'id': node_id,
            'order': order,
            'diameter': D,
            'length': L,
            'area': A,
            'parent': parent_id,
            'children': []
        }
        
        if order > 1:
            left_child = node_generator(order - 1, node_id)
            right_child = node_generator(order - 1, node_id)
            tree[node_id]['children'] = [left_child, right_child]
        
        return node_id
    
    node_generator(max_order)
    return tree

#03_3D_Tree_Generation

def three_d_positions(tree, node_id = 0, position=np.array([0,0,0]), direction=np.array([0, 0, -1]), angle = np.deg2rad(55), branch_factor = 1.0, positions = None): 
#branch factor = vertical length from parent to child
    
    if positions is None:
        positions = {}
        
    positions[node_id] = position
    children = tree[node_id]['children']
    if not children:
        return positions
    
    order = tree[node_id]['order']
    length = tree[node_id]['length']
    
    if np.allclose(direction, [0, 0, 1]) or np.allclose(direction, [0, 0, -1]):
        ortho_dir1 = np.array([1, 0, 0])
    else:
        ortho_dir1 = np.cross(direction, [0, 0, 1])
        ortho_dir1 /= np.linalg.norm(ortho_dir1)
    
    left_direction = (np.cos(angle) * direction + np.sin(angle) * ortho_dir1)
    left_direction /= np.linalg.norm(left_direction)
    right_direction = (np.cos(angle) * direction - np.sin(angle) * ortho_dir1)
    right_direction /= np.linalg.norm(right_direction)
    
    bifurcation_pos = position + direction * length * branch_factor
    
    three_d_positions(tree, children[0], bifurcation_pos, left_direction, angle, branch_factor, positions)
    three_d_positions(tree, children[1], bifurcation_pos, right_direction, angle, branch_factor, positions)
    
    return positions

def build_threed_tree(tree, positions, node_id = 0, tubes = None):
    if tubes is None:
        tubes = []
    
    pos = positions[node_id]
    radius = tree[node_id]['diameter']/2
    
    # sphere = Sphere(pos = pos, r = radius * 1.1, c = 'dodgerblue', alpha = 0.7)
    # tubes.append(sphere)
    
    children = tree[node_id]['children']
    for child_id in children:
        child_pos = positions[child_id]
        direction = child_pos - pos
        height = np.linalg.norm(direction)
        if height == 0:
            continue
        axis = direction/height

        center_pos = pos + axis * height/2
        tube = Cylinder(pos = center_pos, r = radius, height = height, axis = axis, res = 24, c = 'hotpink')
        tubes.append(tube)
        build_threed_tree(tree, positions, child_id, tubes)
    
    return tubes

    
schematic_generation = tree_generator(max_order = 7)
formatted_tree = json.dumps(schematic_generation, indent = 2)
formatted_tree[:1000]

for node_id, data in schematic_generation.items():
    print(f"Node {node_id}: order={data['order']}, children={data['children']}")

positions = three_d_positions(schematic_generation)
tubes = build_threed_tree(schematic_generation, positions)
    
plt = Plotter(title = "Ideal Neonatal Respiratory System (3D)", axes = 1, bg = 'white')
plt.show(tubes, viewup = 'z')

#04_Acoustic_Parameters

#constants
rho = 1.134 #kg/m^3 found using rho = p/RT
air_speed = 343 #m/s

def compute_acoustic_parameters(tree, frequency = 1000):
    omega = 2 * np.pi * frequency
    k = omega/air_speed
    
    for node_id, data in tree.items():
        A = data['area']
        Z0 = rho * air_speed/A #characteristic impedance
        L = data['length']
        T = np.array([
            [np.exp(1j * k * L), 0],
            [0, np.exp(-1j * k * L)]
        ])
        data.update({
            'Z0': Z0,
            'k': k,
            'omega': omega,
            'T': T
        })
    
    return tree

acoustic_tree = compute_acoustic_parameters(schematic_generation, frequency = 1000)