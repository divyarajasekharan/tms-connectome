import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import matplotlib.colors as mcolors
from nilearn import plotting, image, datasets
import nibabel as nib
import numpy as np
import pandas as pd
import sys

def decode_parcel_name(name): 

    components = name.split("_")

    if len(components) == 4: 
        _, hemi, network, num_id = components
        region = ""
    elif len(components) == 5:
        _, hemi, network, region, num_id = components
    else: 
        sys.stderr.write(f"Unable to parse name of parcel: {name}")

    return pd.Series({"hemi" : hemi, "network" : network, "region" : region, "num_id" : num_id})

def plot_connectome_matrix(connectome, node_networks, label_ticks=False):
    # identify boundaries between networks 
    network_boundaries = np.where(node_networks[:-1] != node_networks[1:])[0]    
    network_boundaries = np.insert(network_boundaries, 0, -1)
    network_boundaries = np.append(network_boundaries, 400)
    midpoint = len(network_boundaries[network_boundaries < 200])

    # group parcels across hempispheres by network 
    connectome_group_cols = np.empty(connectome.shape)
    new_boundaries = []

    shift = 0
    for i in range(midpoint-1): 
        idx_left = np.arange(network_boundaries[i]+1, network_boundaries[i+1]+1)
        idx_left_new = np.arange(shift, shift + len(idx_left))
        idx_right = idx_left + 200 
        idx_right_new = idx_left_new + len(idx_left)

        connectome_group_cols[:, idx_left_new] = connectome[:, idx_left]
        connectome_group_cols[:, idx_right_new] = connectome[:, idx_right]

        shift += len(idx_left) + len(idx_right)
        new_boundaries.append(shift)
   
    connectome_group_all = np.empty(connectome.shape)
    shift = 0

    for i in range(midpoint-1): 
        idx_left = np.arange(network_boundaries[i]+1, network_boundaries[i+1]+1)
        idx_left_new = np.arange(shift, shift + len(idx_left))
        idx_right = idx_left + 200 
        idx_right_new = idx_left_new + len(idx_left)

        connectome_group_all[idx_left_new, :] = connectome_group_cols[idx_left, :]
        connectome_group_all[idx_right_new, :] = connectome_group_cols[idx_right, :]

        shift += len(idx_left) + len(idx_right)

    fig, ax = plt.subplots()

    norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=1.0)
    ax_img = plotting.plot_matrix(connectome_group_all, 
                         auto_fit=True, 
                         colorbar=True,
                         norm=norm,
                         figure=fig)
    
    cbar = ax_img.colorbar 
    cbar.set_ticks(np.arange(-0.4, 1.0, 0.2))

    ax.set_xticks(new_boundaries)
    ax.set_yticks(new_boundaries)

    if label_ticks:
        tick_labels = node_networks[network_boundaries[1:midpoint]]
        ax.set_xticklabels(tick_labels, fontsize=5, rotation=-90, horizontalalignment='center')
        ax.set_yticklabels(tick_labels, fontsize=5)
        
    plt.tight_layout()
    plt.savefig("figs/connectome_matrix_unlabeled.svg")

if __name__ == "__main__":
    connectome = np.loadtxt(sys.argv[1], delimiter=',')
    node_ras = pd.read_csv("/oak/stanford/groups/leanew1/atlases/schaefer/" \
        "atlas-Schaefer400/atlas-Schaefer2018_space-MNI152NLin6Asym_res-2_" \
        "desc-400Parcels17Networks_centroid_coords.csv")
    
    color_lut = pd.read_csv("tmp/schaefer_lut.txt", sep="\s+")
    colors = color_lut[["R", "G", "B"]].to_numpy()
    colors = colors / 255

    node_coords = node_ras[["R", "A", "S"]].to_numpy()
    node_networks = node_ras["ROI Name"].map(lambda x: 
                                             decode_parcel_name(x)["network"]).to_numpy()

    s=3 # how many nodes to skip 

    plotting.plot_connectome(connectome[::s, ::s], node_coords[::s, :], 
                            edge_threshold="95%", 
                            node_color=colors[::s, :],
                            edge_cmap='Reds', 
                            node_size=3,
                            node_kwargs = {"alpha" : 0.9},
                            edge_kwargs = {"lw" : .5, "alpha" : 0.5},
                            display_mode='x',
                            colorbar=False,
                            edge_vmin=0,
                            edge_vmax=0.85,
                            output_file="figs/connectome.svg")
    
