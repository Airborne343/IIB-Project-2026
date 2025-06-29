import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx


#01_Scaling
def bronchi_scaling(order, D1 = 0.007, L1 = 0.04, ratio_d = 1.7, ratio_l = 1.3):
    #computing the scaled diameter and length of every generation bronchi and alveolus
    D = D1 * (ratio_d ** (-(order-1)))
    L = L1 * (ratio_l ** (-(order-1)))
    A = np.pi * ((D/2) **2)
    return D, L, A

#02_Geometric_Modelling
def tree_generator(max_order = 15):
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

schematic_generation = tree_generator(max_order = 5)

formatted_tree = json.dumps(schematic_generation, indent = 2)
formatted_tree[:1000]

G = nx.DiGraph()
for node_id, data in schematic_generation.items():
    label = f"{data['order']}"
    G.add_node(node_id, label = label)
    for child in data['children']:
        G.add_edge(node_id, child)
        
pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
# plt.figure(figsize=(10,6))
# nx.draw(G, pos, with_labels=False, arrows=False, node_size=800, node_color="lightblue")
# nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)
# plt.title("Trachea Schematic")
# plt.axis('off')
# plt.show()

#03_Acoustic_Parameters

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

{nid: {
    'order': data['order'],
    'length_m': round(data['length'], 4),
    'Z0 (Pa·s/m³)': round(data['Z0'], 2),
    'k (rad/m)': round(data['k'], 2)
} for nid, data in list(acoustic_tree.items())[:5]}