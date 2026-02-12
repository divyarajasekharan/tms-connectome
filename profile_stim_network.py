import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import sys

sys.path.insert(0, "/oak/stanford/groups/leanew1/users/rdivya/tms/code/utils") 
import get_mask as mask

DEBUG=True

# atlases
ATLAS_DIR="/oak/stanford/groups/leanew1/atlases"
SCHAEFER_DIR=f"{ATLAS_DIR}/schaefer/atlas-Schaefer400"
SCHAEFER_ATLAS=f"{SCHAEFER_DIR}/atlas-Schaefer2018_space-MNI152NLin6Asym_res-2_desc-400Parcels17Networks_dseg.nii.gz"
SCHAEFER_LUT=f"{SCHAEFER_DIR}/atlas-Schaefer2018_space-MNI152NLin6Asym_res-2_desc-400Parcels17Networks_dseg.tsv"

# simulation results 
TMS_DIR="/oak/stanford/groups/leanew1/SHARED_DATASETS/private/tms/bids/data"
XCPD_DIR=f"{TMS_DIR}/derivatives/xcp_d/xcp_d-0.10.7"

DATA_DIR="/oak/stanford/groups/leanew1/users/rdivya/tms/data"
SIM_DIR=f"{DATA_DIR}/simulation-outputs"

networks = ["ContA", "ContB", "ContC",
            "DefaultA", "DefaultB", "DefaultC", 
            "DorsAttnA", "DorsAttnB", "LimbicA", "LimbicB", 
            "SalVentAttnA", "SalVentAttnB", "SomMotA", "SomMotB", 
            "TempPar", "VisCent", "VisPeri"]

def load_stim_mask(sub, ses, threshold):
    sub_mask_dir = f"{SIM_DIR}/sub-{sub}/ses-{ses}/mni_volumes/"

    sub_mask_fn = f"sub-{sub}_ses-{ses}_space-MNI152NLin6Asym_res-2_threshold-{threshold}_weighted-mask.nii.gz"
    sub_mask_fp = f"{sub_mask_dir}/{sub_mask_fn}"

    if not os.path.isfile(sub_mask_fp):
        sys.stderr.write(f"No simulation results found for {sub}, ses-{ses}.\n")
        return None
    
    stim_mask = nib.load(sub_mask_fp)

    return stim_mask

def load_atlas(seg_fp, lut_fp):
    
    seg_img = nib.load(seg_fp)
    affine = seg_img.affine
    seg_arr = np.array(seg_img.get_fdata())

    seg_lut = pd.read_csv(lut_fp, sep='\t')
    return seg_arr, affine, seg_lut

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

def create_network_atlas(atlas, lut, affine=None, out_fp=None): 
    # modify lut to store network association as separate column 
    lut["network"] = lut["label"].map(lambda x: decode_parcel_name(x)["network"])
    lut.set_index("index", inplace=True, drop=True)

    # assign a scalar index to each unique network label 
    networks = np.unique(lut["network"].to_numpy())

    # construct a mapping from network names to integer labels
    network_map = dict(zip(networks, range(1, len(networks)+1)))     
    mapper = lambda x: network_map[x]

    # create a new atlas containing network labels 
    network_atlas = np.empty(atlas.shape)

    # iterate over each voxel in original atlas and assign network label
    it = np.nditer(atlas, flags=['multi_index'])

    for label in it: 
        idx = it.multi_index

        if label not in lut.index.to_numpy(): 
            network_atlas[idx] = 0
            continue 

        network = lut.loc[label, "network"]
        network_atlas[idx] = mapper(network)

    # store network atlas if debugging
    if out_fp is not None:
        network_img = nib.Nifti1Image(network_atlas, affine)
        nib.save(network_img, out_fp)
    
    return network_atlas, network_map

def get_stim_network_fc(fc_df, sub, ses=0, include_hemi=['RH']): 

    sub_rows = fc_df[fc_df.subject == sub]
    ses_row = sub_rows[sub_rows.session == ses]
    
    row_idx = ses_row.index[0]
    sub_df = ses_row.T 

    sub_df.drop(index = ["subject", "session"], inplace=True)
    sub_df.rename(columns = {row_idx : "parcel_fc"}, inplace = True)

    for var in ["hemi", "network", "region", "num_id"]:
        sub_df[var] = sub_df.index.map(lambda x : decode_parcel_name(x)[var])

    if DEBUG and row_idx == 0: 
        sub_df.to_excel(f"debug/{sub}_parcel_fc.xlsx", index=False)

    network_fc = pd.Series(dtype = np.float64)
    network_fc["subject"] = sub
    network_fc["session"] = ses
    network_fc["hemis"] = ", ".join(include_hemi)

    for network in np.unique(sub_df.network): 
        regions_df = sub_df[sub_df.network == network]
        regions_df = regions_df[np.isin(regions_df.hemi, include_hemi)]
        network_fc[network] = np.nanmean(regions_df["parcel_fc"])
    
    return network_fc 

def get_stim_network_fc_group(fc_df, ses=0, include_hemi=['RH']): 

    subjects = np.unique(fc_df["subject"]) 

    networks = np.unique(fc_df.columns[2:].map(lambda x: decode_parcel_name(x)["network"]))
    network_df = pd.DataFrame(columns = ["subject", "session", "hemis"] + list(networks))
    
    for sub in subjects: 
        sub_network_fc = get_stim_network_fc(fc_df, sub, ses, include_hemi) 

        network_df = pd.concat([network_df, sub_network_fc.to_frame().T], 
                               ignore_index=True)
    
    return network_df 

def plot_stim_network_group(network_df): 
    networks = network_df.loc[:, "ContA":"VisPeri"].columns.to_list()

    fig, axs = plt.subplots(nrows=3, ncols=6, sharex='all')
    for i in range(3):
        for j in range(6):
            ax = axs[i][j]

            idx = i*6 + j 
            
            if idx == len(networks):
                break
            
            network = networks[idx]

            ax.hist(network_df[network], density=False)
            ax.set_xlabel(network)
            ax.set_xlim([0,1])
            #ax.set_xticks([-1, -.5, 0, .5, 1])
    
    fig.set_size_inches(12, 8)
    plt.savefig("figs/stimulation_network_overlap.png")

def get_stim_network_overlap(sub, threshold, network_atlas, network_map, 
                             ses="avg_coil"):
    stim_overlap = pd.Series(data = {"subject" : sub, "session" : ses})
    weighted_overlap = pd.Series(data = {"subject" : sub, "session" : ses})

    # load stimulation mask 
    stim_mask = load_stim_mask(sub, ses, threshold)

    if stim_mask is None:
        return stim_overlap, weighted_overlap
    
    # intersect mask with gray matter segmentation 
    stim_mask = mask.intersect_mask_with_sub_gm(sub, stim_mask)

    mask_arr = np.array(stim_mask.get_fdata())

    # define num. voxels & weights in ROI 
    N = np.count_nonzero(mask_arr)
    stim_weights = mask_arr/np.sum(mask_arr)
    flat_stim_idx = np.flatnonzero(stim_weights)
    flat_weights = stim_weights.take(flat_stim_idx)

    # for each network, compute stimulation overlap
    stim_labels = network_atlas.take(flat_stim_idx)

    for network, label in network_map.items(): 
        overlap = stim_labels == label 

        stim_overlap[network] = np.sum(overlap)/N
        weighted_overlap[network] = np.sum(overlap*flat_weights)

    print(stim_overlap)

    return stim_overlap, weighted_overlap 

def get_stim_network_overlap_group(threshold, seg_fp, lut_fp, ses="avg_coil"): 
    # load atlas and construct network atlas
    atlas, affine, lut = load_atlas(seg_fp, lut_fp)
    network_atlas, network_map = create_network_atlas(atlas, lut, affine=affine, 
                                out_fp="debug/schaefer_reconstructed_networks.nii.gz")
    
    # set up data frame to store results
    networks = list(network_map.keys())
    overlap_df = pd.DataFrame(columns = ["subject", "session"] + networks)
    weighted_df = pd.DataFrame(columns = ["subject", "session"] + networks)

    # iterate over subjects with complete simulations
    for sub_sim_dir in Path(SIM_DIR).glob("sub-*/"):
        sub = sub_sim_dir.name.split("-")[-1]

        sub_overlap, sub_weighted = get_stim_network_overlap(
            sub, threshold, network_atlas, network_map, ses)
        
        overlap_df = pd.concat([overlap_df, sub_overlap.to_frame().T], ignore_index=True)
        weighted_df = pd.concat([weighted_df, sub_weighted.to_frame().T], ignore_index=True)

    return overlap_df, weighted_df

def cluster_kmeans_stim_profiles(network_df, n_clusters=4, metric="fc"): 
    network_df.dropna(inplace=True)

    # set up clustering algorithm with random seed initialization 
    # run 100 iterations to ensure convergence to global minimum 
    clusterer = KMeans(n_clusters = n_clusters, init = 'random', n_init=100)
   
    # fit cluster model & assign cluster labels to subjects
    cluster_labels = clusterer.fit_predict(network_df.loc[:, "ContA":]) 
    print("Number of iterations to reach convergence: ", clusterer.n_iter_)
    network_df["cluster_label"] = cluster_labels

    mean_silhouette = silhouette_score(network_df.loc[:, "ContA":], cluster_labels)
    print("Average silhouette score:", mean_silhouette)

    # reduce dimensions by considering summary metrics
    network_df["ContAvg"] = (network_df["ContA"] + 
                             network_df["ContB"] + 
                             network_df["ContC"])/3
    network_df["ContMax"] = np.max(network_df[["ContA", "ContB", "ContC"]], axis=1)
    network_df["ContTot"] = np.sum(network_df[["ContA", "ContB", "ContC"]], axis=1)

    network_df["DefaultAvg"] = (network_df["DefaultA"] + 
                                network_df["DefaultB"] + 
                                network_df["DefaultC"])/3
    network_df["DefaultMax"] = np.max(network_df[["DefaultA", "DefaultB", "DefaultC"]], axis=1)
    network_df["DefaultTot"] = np.sum(network_df[["DefaultA", "DefaultB", "DefaultC"]], axis=1)

    # summarize cluster statistics 
    cluster_df = network_df.groupby(["cluster_label"], as_index=False).mean()

    # visualize 
    cmap = plt.get_cmap('Pastel1')
    
    # create scatter plots y = default vs x = control summary dimensions  
    for stat in ["Avg", "Max", "Tot"]:
        plt.clf()
        cont = f"Cont{stat}"
        default = f"Default{stat}"

        plt.scatter(network_df[cont], network_df[default], 
                c = cmap(network_df["cluster_label"]), marker='o', linestyle='')
    
        plt.scatter(cluster_df[cont], cluster_df[default], 
                c = cmap(cluster_df["cluster_label"]), marker='*', edgecolors='k', s=50, linestyle='')

        plt.xlabel(f'{stat} stimulation site - control network {metric}')
        plt.ylabel(f'{stat} stimulation site - default mode network {metric}')
        plt.savefig(f"figs/stimulation_network_{metric}_{n_clusters}_kmeans_clusters_by_{stat.lower()}_scatter.png")

    # create bar graph to summarize average network connectivity in each cluster
    plt.clf()
    cluster_df.plot(y = networks,
                    kind = 'bar', 
                    rot = 90,
                    stacked = False, 
                    colormap = 'hsv')
    plt.xlabel('Cluster')
    plt.ylabel(f"Average stimulation site - network {metric}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"figs/stimulation_network_{metric}_{n_clusters}_kmeans_clusters_bar.png")

    return mean_silhouette, network_df

def cluster_pca_stim_profiles(network_df): 
    network_df.dropna(inplace=True)

    pca = PCA()
    pca.fit(network_df.loc[:, "ContA":])

    print(pca.explained_variance_ratio_)

    network_df["principal_component"] = pca.transform(network_df.loc[:, "ContA":])[:, 0]

    return network_df

def plot_silhouette(network_df, n_clusters=range(2,11), metric='fc'): 
    silhouette_means = np.empty(len(n_clusters))

    for i, n in enumerate(n_clusters):
        s, _ = cluster_kmeans_stim_profiles(network_df, n_clusters = n, metric=metric)
        silhouette_means[i] = s

    plt.clf()
    plt.plot(n_clusters, silhouette_means, marker='o', linestyle='-')
    plt.ylim([-1, 1])
    plt.savefig("figs/cluster_silhouette_plot.png")

def get_network_segregation(ses_list=['00'], within_networks=['ContA', 'ContB', 'ContC'], 
                            without_networks=['DefaultA', 'DefaultB', 'DefaultC']):

    # get the name of all subjects preprocessed through XCP-D
    sub_dirs = list(Path(XCPD_DIR).glob("sub-*/"))
    subjects = [dir.name.split("sub-")[1] for dir in sub_dirs if os.path.isdir(dir)]
    subjects = np.unique(subjects)

    # set up output data frame
    segregation_df = pd.DataFrame(data = {"subject" :  np.repeat(subjects, len(ses_list)), 
                                         "session" : np.tile(ses_list, len(subjects))},
                                  columns = ["subject", "session", "within_network_fc", 
                                             "without_network_fc", "between_network_fc", 
                                             "segregation"])

    for idx, row in segregation_df.iterrows():
        sub = row["subject"]
        ses = row["session"]

        matrix_fps = list(Path(XCPD_DIR).glob(f"sub-{sub}/ses-{ses}/func/"
            f"sub-{sub}_*task-rs*_seg-Schaefer400_stat-pearsoncorrelation_relmat.tsv"))
        
        n = 0 
        for fp in matrix_fps:
            n += 1
            fc_df = pd.read_csv(fp, sep="\t", usecols=range(1,401))

            if "networks" not in vars():
                networks = fc_df.columns.map(lambda x: decode_parcel_name(x)["network"])

                within_mask = [network in within_networks for network in networks]
                without_mask = [network in without_networks for network in networks]

                within_idx = np.flatnonzero(within_mask)
                without_idx = np.flatnonzero(without_mask)

            fc_matrix = fc_df.to_numpy(dtype = float)

            # select within-network connections
            within_matrix = fc_matrix[within_idx][:, within_idx]

            # mask out elements at and above diagonal to avoid double counting 
            within_matrix = np.tril(within_matrix, -1)

            # select within-network connections for out network 
            without_matrix = fc_matrix[without_idx][:, without_idx]
            without_matrix = np.tril(without_matrix, -1)

            # select connections from in-network to out-network 
            between_matrix = fc_matrix[within_idx][:, without_idx]

            # compute average FC within network versus between networks
            
            # only retain non-zero (i.e., non-masked) elements  
            within_fc = np.nanmean(within_matrix[within_matrix != 0])
            without_fc = np.nanmean(without_matrix[without_matrix != 0])
            between_fc = np.nanmean(between_matrix)

            if n == 1:
                row[["within_network_fc", "without_network_fc", 
                    "between_network_fc"]] = np.array([within_fc, without_fc, 
                                                        between_fc])
            else: 
                row[["within_network_fc", "without_network_fc", 
                    "between_network_fc"]] += np.array([within_fc, without_fc, 
                                                        between_fc])
            
        if not np.isnan(row["between_network_fc"]):
            row[["within_network_fc", "without_network_fc", 
                 "between_network_fc"]] /= n 
            row["segregation"] = (row["within_network_fc"] - 
                                  row["between_network_fc"])/row["within_network_fc"]
        else: 
            row["segregation"] = np.nan

    
        segregation_df.iloc[idx] = row

    return segregation_df

if __name__ == "__main__": 

    fc_df = pd.read_excel(sys.argv[1])

    network_df = get_stim_network_fc_group(fc_df, include_hemi=['LH', 'RH'])
    network_df.to_excel(sys.argv[2], 
                        index=False)

    segregation_df = get_network_segregation(within_networks=['ContA'], 
                                             without_networks=['DefaultA'], 
                                             ses_list=['00'])
    segregation_df.to_excel(sys.argv[3],
                            index=False)