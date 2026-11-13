import ants
import argparse
import nibabel as nib
import numpy as np 
import os
from pathlib import Path
from statistics import mode
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

import sys
#from templateflow import api as tf 

sys.path.insert(0, "/oak/stanford/groups/leanew1/users/rdivya/tms/code/utils") 
import get_mask as mask

TMS_DIR="/oak/stanford/groups/leanew1/SHARED_DATASETS/private/tms/bids/data"
FS_DIR = f"{TMS_DIR}/derivatives/fmriprep-20.2.3/freesurfer"
XCPD_DIR=f"{TMS_DIR}/derivatives/xcp_d/xcp_d-0.10.7"

DATA_DIR="/oak/stanford/groups/leanew1/users/rdivya/tms/data"
ATLAS_DIR=f"{DATA_DIR}/atlases"
MASK_DIR=f"{DATA_DIR}/masks"
SCHAEFER_ATLAS_SURF=f"{ATLAS_DIR}/Schaefer/fsaverage/lh.Schaefer2018_400Parcels_17Networks_order.annot"

SITE_MAP = {'1991' : 'PA', 
            '1976' : 'PV', 
            '2000' : 'MN', 
            '1984' : 'WJ', 
            '1992' : 'MY'
}

MODEL_NAME = {"lm_tx_model" : "LinearTreatmentModel", 
            "lme_tx_model" : "LMETreatmentModel", 
            "lme_fc_tx_model" : "FCChangeModel"}

NETWORK_MAP = {"ContA" : "control", 
               "ContB" : "control", 
               "ContC" : "control", 
               "DefaultA" : "default", 
               "DefaultB" : "default", 
               "DefaultC" : "default", 
               "SalVentAttnA" : "salience",
               "SalVentAttnB" : "salience", 
               "DorsAttnA" : "dorsattn", 
               "DorsAttnB" : "dorsattn", 
               "SomMotA" : "somatomotor", 
               "SomMotB" : "somatomotor",
               "LimbicA" : "limbic", 
               "LimbicB" : "limbic",  
               "VisPeri" : "visual", 
               "VisCent" : "visual", 
               "TempPar" : "temppar"}

NETWORK_ID = {"control" : 0, 
              "default" : 1,
              "salience" : 2,
              "dorsattn" : 3, 
              "somatomotor" : 4, 
              "limbic" : 5, 
              "visual" : 6, 
              "temppar" : 7}

NETWORKS = np.unique(list(NETWORK_MAP.values()))

def redcap_to_bids(redcap_id):
    site_id, sub_id = redcap_id.split("-")

    return "TMS" + SITE_MAP[site_id] + sub_id.zfill(3)
    
def create_functional_adjacency_matrix(fc_threshold=0.3, percentile=None, 
                                       task=None, sub_list=None): 
    if task is None: 
        matrices = list(Path(XCPD_DIR).glob("sub-*/ses-*/func/"
        "*_seg-Schaefer400_stat-pearsoncorrelation_relmat.tsv"))
    else:
        matrices = list(Path(XCPD_DIR).glob("sub-*/ses-*/func/"
            f"*task-{task}*_seg-Schaefer400_stat-pearsoncorrelation_relmat.tsv"))
    
    matrices = np.array(matrices)

    if sub_list is not None:
        get_subject = lambda x : x.split("sub-")[1].split("_")[0] 
        subjects = [get_subject(matrix.name) for matrix in matrices]

        subjects_in_list = np.isin(subjects, sub_list)

        matrices = matrices[subjects_in_list]

    print("Number of subject-level matrices to average: ", len(matrices))

    sum_matrix = pd.read_csv(matrices[0], sep="\t", 
                             usecols=range(1,401)).to_numpy(dtype=float)

    # store number of non-nan values summed in each cell
    n = np.isfinite(sum_matrix)
    print(n)

    # compute sum of all observed pairwise fcs between regions 
    # and store in sum_matrix
    for matrix in matrices[1:]:
        matrix = pd.read_csv(matrix, sep="\t", 
                             usecols=range(1,401)).to_numpy(dtype=float)
        
        sum_matrix = np.nansum(np.dstack((sum_matrix, matrix)), axis=2)
        n = np.nansum(np.dstack((n, np.isfinite(matrix))), axis=2)
    
    # compute mean pairwise fcs by dividing sum_matrix by number of observations, n
    fc_matrix = sum_matrix / n 
    np.savetxt(f"tmp/schaefer400_functional_connectome_mat.csv", 
               fc_matrix, 
               delimiter=',')

    np.savetxt(f"tmp/schaefer400_nobs_mat.csv", 
            n, 
            delimiter=',')

    # if using percentile to threshold edges, define threshold 
    if percentile is not None: 
        fc_threshold = np.percentile(fc_matrix, percentile)

        print(f"Functional connectivity threshold for top {percentile}% of edges: {fc_threshold}")

    # compute adjacency matrix by preserving all connections with fc >= threshold
    adj_matrix = np.array(fc_matrix >= fc_threshold, dtype=int)

    return adj_matrix

""" Use surface atlas to define adjacency matrix by identifying neighboring 
clusters in Schaefer atlas. """
def create_adjacency_matrix(nparcels=400): 

    # include a meaningless 0-index row/column 
    # for convenient indexing with 1-indexed atlas labels
    W_mat = np.zeros((nparcels+1, nparcels+1))

    for hemi in ['lh', 'rh']:
        # load schaefer atlas 
        atlas_fp =f"{ATLAS_DIR}/Schaefer/fsaverage/{hemi}.Schaefer2018_{nparcels}Parcels_17Networks_order.annot"
        labels, _, _ = nib.freesurfer.io.read_annot(atlas_fp)

        # load fsaverage template 
        template_fp = f"{FS_DIR}/fsaverage/surf/{hemi}.inflated"
        template = nib.freesurfer.io.read_geometry(template_fp)
        _, triangles = template[0:2]
        
        # TO-DO: seriously need to look into why fsaverage parcels numbering restarts per hemishphere 
        if hemi == "rh": 
            labels = labels + int(nparcels/2)

        # extract parcel boundaries 
        bdry_tri = np.array([not np.all(labels[tri] == labels[tri[0]]) 
                            for tri in triangles]).astype(bool)
        
        bdry_tri = triangles[bdry_tri]

        for tri in bdry_tri: 
            unique_labels = np.unique(labels[tri])

            for label in unique_labels:
                W_mat[label, unique_labels] = 1 
        
    # remove zeroeth row/column
    W_mat = W_mat[1:, 1:]

    # sanity checks 
    assert(np.all(np.diag(W_mat) == 1))

    return W_mat     

def load_adjacency_matrix(fc_adj=False, add_spatial=False, fc_threshold=None, 
                          percentile=None, task=None, sub_list=None, 
                          overwrite=False): 
    # load adjacency matrix or create if not found
    if not fc_adj or add_spatial:
        W_adj_fp = "tmp/schaefer400_spatial_adjacency_mat.csv"

        if os.path.isfile(W_adj_fp) and not overwrite:
            W_adj = np.loadtxt(W_adj_fp, delimiter=',', dtype=int)
        else:
            W_adj = create_adjacency_matrix()
            np.savetxt(W_adj_fp, W_adj, fmt='%d', delimiter=',')

    if fc_adj: 
        if percentile is not None: 
            W_adj_fp = f"tmp/schaefer400_functional_adjacency_percentile{percentile}"
        else:
            W_adj_fp = f"tmp/schaefer400_functional_adjacency_FC{fc_threshold}"

        if task is not None: 
            W_adj_fp += f"_task-{task}.csv"
        else:
            W_adj_fp += "_task-all.csv"

        if os.path.isfile(W_adj_fp) and not overwrite:
            W_adj_fc = np.loadtxt(W_adj_fp, delimiter=',', dtype=int)
        else:
            W_adj_fc = create_functional_adjacency_matrix(fc_threshold=fc_threshold, 
                                                       percentile=percentile, 
                                                       task = task, sub_list=sub_list)
            np.savetxt(W_adj_fp, W_adj_fc, fmt='%d', delimiter=',')

        if add_spatial:
            W_adj_fp = W_adj_fp.split("_mat.csv")[0] + "_plus_spatial_mat.csv"

            if os.path.isfile(W_adj_fp):
                 W_adj = np.loadtxt(W_adj_fp, delimiter=',', dtype=int)
            else:
                W_adj = np.logical_or(W_adj_fc, W_adj)
                np.savetxt(W_adj_fp, W_adj, fmt='%d', delimiter=',')
        else: 
            W_adj = W_adj_fc 

    print("Adjacency matrix stored at: ", W_adj_fp)

    return W_adj

""" Get parcel size in voxels to scale cluster mass. """
def get_parcel_size(nparcels=400, gm_threshold=0.3): 
    sizes = np.zeros(nparcels)

    atlas_fp = f"{ATLAS_DIR}/Schaefer/mni/Schaefer2018_{nparcels}Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"

    ants_img = ants.image_read(atlas_fp, reorient = 'LPI')
    atlas_arr = ants_img.numpy()

    # load average gray matter mask 
    gm_mask_img = mask.get_avg_gm_mask(template='MNI152NLin6Asym')
    gm_mask_arr = np.array(gm_mask_img.get_fdata())

    assert(np.all(np.diag(gm_mask_img.affine) > 0))
    
    atlas_arr[gm_mask_arr < gm_threshold] = 0

    for i in range(nparcels):
        label = i+1
        sizes[i] = np.count_nonzero(atlas_arr == label)
    
    return sizes

def load_parcel_size():
    psize_fp = "tmp/schaefer400_parcel_sizes.csv" 

    if os.path.isfile(psize_fp):
        parcel_size = np.loadtxt(psize_fp, delimiter=',', dtype=int)
    else:
        parcel_size = get_parcel_size()
        np.savetxt(psize_fp, parcel_size, fmt='%d', delimiter=',')

    return parcel_size

def get_clusters(results_df, effect, W_adj, parcel_size, outcomes, alpha = 0.05, 
                 F_stat=False, maintain_direction=True, verbose=False):

    # set up cluster df 
    cluster_df = pd.DataFrame(columns = ["outcome", "cluster_num", "parcel_ids", 
                                "parcel_labels", "cluster_network", "cluster_psize", 
                                "cluster_vsize", "cluster_mass"])
    
    for outcome in outcomes:
        if verbose:
            print("Outcome: ", outcome, "\n", "-"*80)

        outcome_df = results_df[results_df.outcome == outcome]
        outcome_df = outcome_df.iloc[0:400]

        p_values = outcome_df[f"p_{effect}"].to_numpy()

        parcels = outcome_df["parcel"].to_numpy()
        nparcels = len(parcels)
        parcel_ids = np.arange(nparcels)

        if F_stat:
            test_values = outcome_df[f"F_{effect}"].to_numpy()
            W_adj_dir = W_adj
        else:
            test_values = outcome_df[f"t_{effect}"].to_numpy()
            
            if maintain_direction: 
                W_adj_dir = np.copy(W_adj)

                where_pos = test_values > 0 

                row_pos = np.tile(where_pos[np.newaxis].T, (1, nparcels))
                col_pos = np.tile(where_pos, (nparcels, 1))

                dir_conflict = np.logical_xor(row_pos, col_pos)

                W_adj_dir[dir_conflict] = 0

            else: 
                W_adj_dir = W_adj

        ### identify clusters of significant parcels ###
        clusters = []

        # create a list of parcels to visit 
        # mark significant parcels as to-be-visited
        to_visit = np.full(nparcels, False)
        to_visit[p_values < alpha] = True

        while sum(to_visit) > 0:
            if verbose:
                print("Cluster ID: ", len(clusters), "\n", "-"*80)
                print("Remaining parcels to visit: ", parcel_ids[to_visit])

            # run BFS to identify a cluster of significant parcels
            cluster = []

            # add a parcel to be visited to the queue & mark as visited
            queue = []
            p = parcel_ids[to_visit][0]
            queue.append(p)
            to_visit[p] = False

            # BFS algorithm
            iter=0
            while len(queue) > 0: 
                if verbose:
                    print("\tBFS iteration: ", iter)

                iter += 1

                # dequeue the first parcel & add to the cluster 
                p = queue.pop(0)
                cluster.append(p)

                if verbose:
                    print("\t\tVisiting: ", p)

                # search for neighboring significant parcels that have not been 
                # visited 
                if verbose:
                    print("\t\tNeighbors: ", parcel_ids[W_adj_dir[p] > 0])
                for n in parcel_ids[W_adj_dir[p] > 0]:
                    if to_visit[n]:
                        queue.append(n)
                        to_visit[n] = False
                
            clusters.append(cluster)

        if verbose:
            print("Final clusters: ", clusters)

        ### quantify cluster size and cluster mass ###
        cluster_ids = []
        cluster_labels = []
        cluster_psize = []
        cluster_vsize = []
        cluster_mass = []

        for cluster in clusters: 
            psize = len(cluster) # cluster size in terms of num parcels
            
            vsize = np.sum(parcel_size[cluster]) # cluster size in terms of num voxels
            mass = np.sum(parcel_size[cluster] * test_values[cluster]) 
            
            #mass = np.sum(test_values[cluster])

            cluster_vsize.append(vsize)
            cluster_psize.append(psize)
            cluster_mass.append(mass)

            sep = ", "
            cluster_ids.append(np.array(cluster) + 1) # 1-index labels to match Schaefer atlas
            cluster_labels.append(sep.join(parcels[cluster]))
    
        # add permutation clusters for outcome to cluster_df
        outcome_df = pd.DataFrame(data = {"outcome" : outcome, 
                                        "cluster_num" : np.arange(len(clusters)),
                                        "parcel_ids" : cluster_ids,
                                        "parcel_labels" : cluster_labels, 
                                        "cluster_psize" : cluster_psize, 
                                        "cluster_vsize" : cluster_vsize, 
                                        "cluster_mass" : cluster_mass})
        
        cluster_df = pd.concat([cluster_df, outcome_df])
    
    ### assign network to cluster based on majority schaefer assignment ### 
    cluster_df = assign_cluster_network(cluster_df)

    return cluster_df 

def get_network(parcel_labels): 
    parcel_labels = parcel_labels.split(", ") 

    networks = [NETWORK_MAP[label.split("_")[2]] for label in parcel_labels]

    majority = mode(networks)
    
    return majority 

def assign_cluster_network(cluster_df):
    cluster_df["cluster_network"] = cluster_df["parcel_labels"].map(get_network)

    return cluster_df

def get_max_clusters(cluster_df, outcomes, per_network=False): 

    ### extract maximal cluster stats per outcome ### 
    cluster_stats = []
    for outcome in outcomes: 
        if per_network:
            for network in NETWORKS:
                outcome_df = cluster_df[np.logical_and(cluster_df.outcome == outcome,
                                        cluster_df.cluster_network == network)]

                if len(outcome_df) > 0:
                    max_cluster_psize = np.max(outcome_df["cluster_psize"])
                    max_cluster_vsize = np.max(outcome_df["cluster_vsize"])
                    max_cluster_mass = np.take(outcome_df["cluster_mass"].to_numpy(), 
                                    np.argmax(np.abs(outcome_df["cluster_mass"])))
                else: 
                    max_cluster_psize, max_cluster_vsize, max_cluster_mass = 0, 0, 0

                cluster_stats.append([max_cluster_psize, max_cluster_vsize, max_cluster_mass])
        else: 
            outcome_df = cluster_df[cluster_df.outcome == outcome]

            if len(outcome_df) > 0:
                max_cluster_psize = np.max(outcome_df["cluster_psize"])
                max_cluster_vsize = np.max(outcome_df["cluster_vsize"])
                max_cluster_mass = np.take(outcome_df["cluster_mass"].to_numpy(), 
                                np.argmax(np.abs(outcome_df["cluster_mass"])))
            else: 
                max_cluster_psize, max_cluster_vsize, max_cluster_mass = 0, 0, 0

            cluster_stats.append([max_cluster_psize, max_cluster_vsize, max_cluster_mass])

    return cluster_stats

def get_cluster_pvalues(cluster_df, null_df, outcomes, per_network=False,
                        stats=["psize", "vsize", "mass"], one_sided=False):


    for stat in stats: 
            cluster_df[f"p_perm_{stat}"] = np.nan 

    for outcome in outcomes: 
        if per_network: 
            for network in NETWORKS:
                outcome_df = null_df[np.logical_and(null_df.outcome == outcome,
                                        null_df.cluster_network == network)]

                for stat in stats: 
                    null_values = outcome_df[f"max_cluster_{stat}"].to_numpy()
                    
                    if one_sided: 
                        get_p = lambda x : np.sum(null_values > x) / len(null_values) if x >= 0 else np.sum(null_values < x) / len(null_values)
                    else:
                        get_p = lambda x : np.sum(np.abs(null_values) > np.abs(x)) / len(null_values)

                    which_clusters = np.logical_and(cluster_df.outcome == outcome, 
                                        cluster_df.cluster_network == network)
                    
                    cluster_df.loc[which_clusters, f"p_perm_{stat}"] = \
                        cluster_df.loc[which_clusters, f"cluster_{stat}"].map(get_p)              
        else: 
            outcome_df = null_df[null_df.outcome == outcome]

            for stat in stats: 
                null_values = outcome_df[f"max_cluster_{stat}"].to_numpy()
                
                if one_sided: 
                    get_p = lambda x : np.sum(null_values > x) / len(null_values) if x >= 0 else np.sum(null_values < x) / len(null_values)
                else:
                    get_p = lambda x : np.sum(np.abs(null_values) > np.abs(x)) / len(null_values)

                cluster_df.loc[cluster_df.outcome == outcome, f"p_perm_{stat}"] = \
                    cluster_df.loc[cluster_df.outcome == outcome, 
                                f"cluster_{stat}"].map(get_p)              
    
    return cluster_df

def generate_parser():
    parser = argparse.ArgumentParser(prog="cluster_efield_effects", 
            description="Cluster effects from whole-brain models.")
    
    # required args
    parser.add_argument("clinical_df", 
                        help = "File path to workbook containing all outcome variables & covariates.")
    parser.add_argument("parcel_df",
                        help = "Workbook containing values of fixed effects variables for atlas parcels.")
    parser.add_argument("out_fp", help = "Destination file path to store results.")
    
    # optional arguments  
    parser.add_argument("--outcomes", nargs="+", default=["qids_tot", "pcl5_tot", 
                                                          "vr36_emo_wellbeing", 
                                                          "vr36_energy", 
                                                          "vr36_role_limit_emo", 
                                                          "vr36_soc_fx"],
                         help = "Names of variables for which to cluster effects")

    # options for clustering results
    parser.add_argument("--fc_adj", action='store_true', 
        help="Use a functional adjacency matrix based on subject-averaged connectome.")
    parser.add_argument("--task", default='rs', help="Task to use to create connectome")
    parser.add_argument("--fc_threshold", default=0.3, type=float,  
        help="Functional connectivity threshold at which nodes are considered adjacent.")
    parser.add_argument("--percentile", type=float, 
                         help = "Top percentile of edges to retain in adjacency matrix. " \
                         "Alternative to setting an absolute threshold.")
    parser.add_argument("--overwrite", action='store_true', 
                        help="Whether to overwrite stored connectome (if found).")
    parser.add_argument("--sub_list", 
        help="List of subjects to use in calculation of group-averaged connectome.")
    parser.add_argument("--add_spatial", action='store_true', 
                        help = "Add spatially adjacent edges to adjacency matrix")

    # options for clustering
    parser.add_argument('--per_network', action="store_true", help="Classify clusters by network.")
    parser.add_argument("--maintain_direction", action='store_true', 
                        help="Whether the direction of the effect should be kept " \
                        "consistent within clusters formed")
    
    # modes of running this program 
    parser.add_argument("--cluster_effects", action='store_true', 
        help="Analyze experimental results to identify clusters of significant results")
    parser.add_argument("--get_pvalues", action='store_true', 
        help="Cross reference experimentally observed clusters and simulated null values to assign p-values")

    # optional data frames to input for analysis
    parser.add_argument("--results_df", help="Data frame containing model results to form clusters")
    parser.add_argument("--cluster_df", help="Data frame describing clusters of significant parcels")
    parser.add_argument('--null_df', help="Reference distribution to quantify p-values")
    
    # options for effects testing 
    parser.add_argument('--alpha', default=0.05, type=float, help="Significance threshold")
    parser.add_argument('--effect', default='interaction_session2_parcel', help="Main effect of interest")
    parser.add_argument('--F_stat', action='store_true', help="Whether to quantify effect using F-statistic")
    parser.add_argument('--one_sided', action='store_true', help="Whether to conduct a two-sided hypothesis test")

    return parser

if __name__ == "__main__": 
    parser = generate_parser()
    args = parser.parse_args()

    # load data sets 
    clinical_df = pd.read_csv(args.clinical_df)
    clinical_df["subject"] = clinical_df["record_id"].map(redcap_to_bids)

    parcel_df = pd.read_excel(args.parcel_df)

    # load adjacency matrix or create if not found
    sub_list = None
    if args.sub_list is not None: 
        print(args.sub_list)
        if not os.path.isfile(args.sub_list): 
            print(f"No file found at {args.sub_list}!")
        else:        
            sub_list = pd.read_excel(args.sub_list)
            sub_list = sub_list["subject"].to_numpy()

    W_adj = load_adjacency_matrix(args.fc_adj, args.add_spatial, 
                                  args.fc_threshold, args.percentile, 
                                  task = args.task, sub_list=sub_list, 
                                  overwrite = args.overwrite)
    
    # load parcel sizes or calculate if not found
    psize_fp = "tmp/schaefer400_parcel_sizes.csv" 

    if os.path.isfile(psize_fp):
        psize = np.loadtxt(psize_fp, delimiter=',', dtype=int)
    else:
        psize = get_parcel_size()
        np.savetxt(psize_fp, psize, fmt='%d', delimiter=',')

    cluster_df = None
    if args.cluster_effects:
        results_df = pd.read_excel(args.results_df) 
        cluster_df = get_clusters(results_df, args.effect, W_adj, psize, args.outcomes, 
                                  F_stat=args.F_stat, alpha=args.alpha, 
                                  maintain_direction=args.maintain_direction)

        cluster_df.to_excel(args.out_fp, index=False)

    if args.get_pvalues:
        if cluster_df is None and args.cluster_df is not None:
            cluster_df = pd.read_excel(args.cluster_df)
        else: 
            sys.stderr.write("No cluster dataframe provided!")
            exit(1)

        if args.null_df is not None:
            null_df = pd.read_excel(args.null_df)
        else: 
            sys.stderr.write("No null statistics provided!")
            exit(1)

        cluster_df = get_cluster_pvalues(cluster_df, null_df, args.outcomes, 
                                         per_network = args.per_network, 
                                         one_sided=args.one_sided)

        cluster_df.to_excel(args.out_fp, index=False)