"""11_figure6_tension_network.py: Figure 6 - network of the tensions between Planck, NVSS, RACS-low and CatWISE.

Fast (seconds). Reads outputs/nre/{first}_{second}/stats/ensemble_stats.npy
(05_ensemble.py) for the six kinematic combinations (CatWISE with the ecliptic bias
correction), and writes outputs/figures/figure6.pdf. Each edge is labelled with the
ensemble tension T and its 1 sigma error.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import scienceplots
import config
if not os.path.exists(config.FIGURES):
    os.makedirs(config.FIGURES)
plt.style.use('science')

# ---------------------------------------------------------------------------
# Ensemble tension of each combination, formatted as an edge label
# ---------------------------------------------------------------------------
def tension_label(FIRST_dataset, SECOND_dataset):
    BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)
    ensemble_stats = np.load(f'{BASE_DIR}stats/ensemble_stats.npy', allow_pickle=True)
    ensemble_sigmaD = np.array(ensemble_stats[0])
    T = ensemble_sigmaD[:, 0].mean()
    T_lower_error = np.sqrt(np.sum((ensemble_sigmaD[:, 0] - ensemble_sigmaD[:, 1])**2))
    T_upper_error = np.sqrt(np.sum((ensemble_sigmaD[:, 2] - ensemble_sigmaD[:, 0])**2))
    if np.round(float(T_upper_error), 2) == np.round(float(T_lower_error), 2):
        return rf'${float(T):.2f}\pm{float(T_upper_error):.2f}\sigma$'
    return rf'${float(T):.2f}\substack{{+{float(T_upper_error):.2f}\\-{float(T_lower_error):.2f}}}\sigma$'

# ---------------------------------------------------------------------------
# Build and draw the network (Planck in the centre, the surveys around it),
# drawn as in the tension network of Land-Strykowski et al. (2025; dipole-tensions)
# ---------------------------------------------------------------------------
BOX_W = 0.5   # width  of the CatWISE / NVSS / RACS-low boxes (data coordinates)
BOX_H = 0.13  # height of the CatWISE / NVSS / RACS-low boxes (data coordinates)
fontsize = 11

nodes = ['CatWISE', 'NVSS', 'RACS-low', r'\textit{Planck}']
pos = {
    'CatWISE': (0, 1.0),
    'NVSS': (-np.sqrt(3)/2, -0.5),
    'RACS-low': (np.sqrt(3)/2, -0.5),
    r'\textit{Planck}': (0, 0.0),
}

# Create the graph, with each edge labelled by the ensemble tension
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edge('CatWISE', 'NVSS', weight=tension_label('nvss', 'catwise'))
G.add_edge('CatWISE', 'RACS-low', weight=tension_label('racs', 'catwise'))
G.add_edge('NVSS', 'RACS-low', weight=tension_label('racs', 'nvss'))
G.add_edge('CatWISE', r'\textit{Planck}', weight=tension_label('planck', 'catwise'))
G.add_edge('NVSS', r'\textit{Planck}', weight=tension_label('planck', 'nvss'))
G.add_edge('RACS-low', r'\textit{Planck}', weight=tension_label('planck', 'racs'))

plt.figure(figsize=(3.5, 3))

# Survey nodes: edges only here (the rectangular boxes and labels are added below)
square_nodes = ['CatWISE', 'NVSS', 'RACS-low']
nx.draw(
    G, pos, nodelist=square_nodes, with_labels=False, node_color='white', node_size=0,
    edgecolors='black', linewidths=1, font_color='black', node_shape='s', font_size=fontsize
)

# Planck: circular node in the centre
nx.draw(
    G, pos, nodelist=[r'\textit{Planck}'], with_labels=True, node_color='white', node_size=int(1000*1.75),
    edgecolors='black', linewidths=1, font_color='black', node_shape='o', font_size=fontsize
)

# Edge labels (the tensions)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)},
    font_size=fontsize
)

# Survey nodes: rectangular boxes with the survey names, drawn on top of the edges
ax = plt.gca()
for node in square_nodes:
    x, y = pos[node]
    rect = mpatches.FancyBboxPatch(
        (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H, boxstyle='square,pad=0',
        linewidth=1, edgecolor='black', facecolor='white', zorder=5, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(x, y, node, ha='center', va='center', fontsize=fontsize, color='black', zorder=6, clip_on=False)

plt.savefig(config.FIGURES + '/figure6.pdf', dpi=300, bbox_inches="tight")
