import ants
import argparse
from datetime import datetime
import multiprocessing
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
from model_effects import run_lm_tx_model, run_lme_tx_model, run_lme_fc_tx_model_2

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

def get_clusters(results_df, effect, W_adj, parcel_size, outcomes, alpha = 0.05, 
                 F_stat=False, maintain_direction=True):

    # set up cluster df 
    cluster_df = pd.DataFrame(columns = ["outcome", "cluster_num", "parcel_ids", 
                                "parcel_labels", "cluster_network", "cluster_psize", 
                                "cluster_vsize", "cluster_mass"])
    
    for outcome in outcomes:
        print("Outcome: ", outcome, "\n", "-"*80)
        outcome_df = results_df[results_df.outcome == outcome]
        outcome_df = outcome_df.iloc[0:400]

        p_values = outcome_df[f"p_{effect}"].to_numpy()

        if F_stat:
            test_values = outcome_df[f"F_{effect}"].to_numpy()
            W_adj_dir = W_adj
        else:
            test_values = outcome_df[f"t_{effect}"].to_numpy()
            
            print(len(test_values))

            if maintain_direction: 
                W_adj_dir = np.copy(W_adj)

                where_pos = test_values > 0 

                row_pos = np.tile(where_pos[np.newaxis].T, (1, 400))
                col_pos = np.tile(where_pos, (400, 1))

                dir_conflict = np.logical_xor(row_pos, col_pos)

                W_adj_dir[dir_conflict] = 0

            else: 
                W_adj_dir = W_adj

        parcels = outcome_df["parcel"].to_numpy()
        nparcels = len(parcels)
        parcel_ids = np.arange(nparcels)

        ### identify clusters of significant parcels ###
        clusters = []

        # create a list of parcels to visit 
        # mark significant parcels as to-be-visited
        to_visit = np.full(nparcels, False)
        to_visit[p_values < alpha] = True

        while sum(to_visit) > 0:
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
                print("\tBFS iteration: ", iter)
                iter += 1

                # dequeue the first parcel & add to the cluster 
                p = queue.pop(0)
                cluster.append(p)
                print("\t\tVisiting: ", p)

                # search for neighboring significant parcels that have not been 
                # visited 
                print("\t\tNeighbors: ", parcel_ids[W_adj_dir[p] > 0])
                for n in parcel_ids[W_adj_dir[p] > 0]:
                    if to_visit[n]:
                        queue.append(n)
                        to_visit[n] = False
                
            clusters.append(cluster)

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

def permute_data(data_df, on="subject", permute=False):

    # identify unique subject identifiers
    subjects = np.unique(data_df["subject"]) 

    if on == "subject": 
        # initialize numpy random number generator
        rng = np.random.default_rng()

        # use rng to shuffle subject labels
        shuffled_subjects = subjects.copy()
        rng.shuffle(shuffled_subjects)

        shuffle_mapping = dict(zip(subjects, shuffled_subjects))

        # re-assign shuffled subject labels, repeating for every session
        data_df["subject"] = data_df["subject"].map(lambda x : shuffle_mapping[x])

    elif on == "session": 

        for sub in subjects:
            # re-initialize numpy random number generator
            rng = np.random.default_rng()
            
            # extract parcellated fc values from each session for subject 
            ses_fcs = data_df.loc[data_df.subject == sub]
            ses_fcs = ses_fcs.iloc[:, 2:]
            sub_idx = ses_fcs.index 

            # shuffle session labels & keep missing data
            if permute: 
                shuffled_fcs = ses_fcs.to_numpy()
                rng.shuffle(shuffled_fcs)

                data_df.iloc[sub_idx, 2:] = shuffled_fcs

            else: # bootstrap
                # maintain missingness structure of data
                keep_idx = ses_fcs.dropna(how = 'all').index.to_numpy()

                ses_fcs = ses_fcs.loc[keep_idx, :].to_numpy()

                # randomly sample sessions with replacement
                shuffled_fcs = rng.choice(ses_fcs, size = len(keep_idx))
                
                # honestly have no idea if the pandas slice is mutable so
                data_df.iloc[keep_idx, 2:] = shuffled_fcs
    
    if "session" in data_df.columns:
        data_df.sort_values(by = ["subject", "session"], inplace=True)
    else:
        data_df.sort_values(by = ["subject"], inplace=True)
    
    return data_df

def run_permutation(clinical_df, parcel_df, outcomes, covariates, 
                    W_adj, parcel_sizes, run_id=0, perm_id=0, 
                    effect="parcelEP", per_network=False, 
                    maintain_direction=False, F_stat=False, 
                    alpha = 0.05): 

    tmp_dir = "/scratch/users/rdivya/tms/tmp/permute"

    ### generate permutation model ###    
    parcel_df = permute_data(parcel_df, on = "subject")

    # fit models to permuted data 
    results_df = run_lme_tx_model(parcel_df, clinical_df, outcomes, 
                                covariates=covariates)
    
    ### store results of permutation to tmp directory ### 

    os.makedirs(tmp_dir, exist_ok=True)

    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H:%M:%S")
    
    results_fn = f"Run-{run_id}_Permutation-{perm_id}_Model-SymptomChange_{date}_{time}.xlsx" 
    results_fp = f"{tmp_dir}/{results_fn}"
    results_df.to_excel(results_fp, index=False)

    ### for each clinical outcome model, identify clusters of significant parcels ###
    
    cluster_df = get_clusters(results_df, effect, W_adj, parcel_sizes, outcomes, 
                              F_stat=F_stat, alpha=alpha, 
                              maintain_direction=maintain_direction)  

    # store cluster results to tmp directory
    cluster_fn = f"Run-{run_id}_Permutation-{perm_id}_Model-SymptomChange_Clusters_{date}_{time}.xlsx" 
    cluster_fp = f"{tmp_dir}/{cluster_fn}"
    cluster_df.to_excel(cluster_fp, index=False)

    ### extract maximal cluster stats per outcome ### 
    cluster_stats = get_max_clusters(cluster_df, outcomes, 
                                     per_network=per_network)

    return cluster_stats

def run_fc_permutation(clinical_df, parcel_df, moderator_df, 
                    outcomes, covariates, moderator, 
                    W_adj, parcel_sizes, run_id=0, perm_id=0, 
                    effect="parcelEP", per_network=False, 
                    maintain_direction=False, shuffle_on = 'session',
                    include_bl=True, F_stat=False, alpha = 0.05): 

    tmp_dir = "/scratch/users/rdivya/tms/tmp/permute"
    #tmp_dir = "debug"

    ### generate permutation model ###    
    if shuffle_on == "session":
        parcel_df = permute_data(parcel_df, on = "session")
    elif shuffle_on == "moderator":
        moderator_df = permute_data(moderator_df, on = "subject")
    else: 
        sys.stderr.write(f"Unrecognized shuffle option: {shuffle_on}.\n")
        return []
        
    # fit models to permuted data 
    results_df = run_lme_fc_tx_model_2(parcel_df, clinical_df, moderator_df, 
                                covariates=covariates, moderator=moderator, 
                                include_bl=include_bl)
    
    ### store results of permutation to tmp directory ### 

    os.makedirs(tmp_dir, exist_ok=True)

    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H:%M:%S")
    
    results_fn = f"Run-{run_id}_Permutation-{perm_id}_Model-FCChange_Moderator-{moderator}_Shuffle-{shuffle_on}_{date}_{time}.xlsx" 
    
    results_fp = f"{tmp_dir}/{results_fn}"
    results_df.to_excel(results_fp, index=False)

    ### for each clinical outcome model, identify clusters of significant parcels ###
    
    cluster_df = get_clusters(results_df, effect, W_adj, parcel_sizes, outcomes, 
                            F_stat=F_stat, alpha=alpha, 
                            maintain_direction=maintain_direction)  

    ### extract maximal cluster stats per outcome ### 
    cluster_stats = get_max_clusters(cluster_df, outcomes, per_network=per_network)

    return cluster_stats

def create_null_distribution(N, clinical_df, parcel_df, outcomes, covariates,
                             W_adj, parcel_sizes, effect='parcelEP', 
                             run_id="Unknown", perm_ids=[], F_stat=False, 
                             alpha=0.05,per_network=False, 
                             maintain_direction=False, nthreads=16): 
    
    n_networks = 1
    if per_network: 
        n_networks = len(NETWORKS)

    if len(perm_ids) != N:
        perm_ids = np.arange(N) 

    null_df = pd.DataFrame(data = {"outcome": np.tile(np.repeat(outcomes,
                                                                n_networks), 
                                                        N), 
                                "permutation" : np.repeat(perm_ids, 
                                                    len(outcomes) * n_networks), 
                                "cluster_network" : "n/a",
                                "effect" : effect, 
                                "alpha" : alpha, 
                                "F-statistic" : F_stat}, 
                                columns = ["outcome", "permutation", "effect", 
                                    "alpha", "F-statistic", "cluster_network", 
                                    "max_cluster_psize", "max_cluster_vsize", 
                                    "max_cluster_mass"])

    if per_network: 
        null_df["cluster_network"] = np.tile(NETWORKS, N * len(outcomes))

    pool = multiprocessing.Pool(nthreads)

    cluster_stats = pool.starmap(run_permutation, zip(
        [clinical_df] * N, 
        [parcel_df] * N, 
        [outcomes] * N, 
        [covariates] * N, 
        [W_adj] * N, 
        [parcel_sizes] * N, 
        np.repeat(run_id, N),
        perm_ids, 
        [effect] * N, 
        [per_network] * N, 
        [maintain_direction] * N, 
        [F_stat] * N, 
        [alpha] * N
    ))
    
    cluster_stats = np.vstack(cluster_stats)
    null_df.iloc[:, 6:] = cluster_stats 

    return null_df

def create_null_fc_distribution(N, clinical_df, parcel_df, outcomes, covariates,
                             W_adj, parcel_sizes, effect, 
                             moderator_df=None, moderator='n/a', 
                             run_id="Unknown", perm_ids=[], 
                             F_stat=False, alpha=0.05,
                             per_network=False, maintain_direction=False, 
                             shuffle_on='session', include_bl=True, 
                             nthreads=16): 
    
    n_networks = 1
    if per_network: 
        n_networks = len(NETWORKS)

    if len(perm_ids) != N:
        perm_ids = np.arange(N) 

    null_df = pd.DataFrame(data = {"outcome" : "parcel_fc",
                            "moderator": moderator, 
                            "permutation" : np.repeat(perm_ids, n_networks), 
                            "cluster_network" : "n/a",
                            "effect" : effect, 
                            "alpha" : alpha, 
                            "F-statistic" : F_stat}, 
                        columns =["outcome", "moderator", "permutation", 
                                   "effect", "alpha", "F-statistic", 
                                   "cluster_network", 
                                    "max_cluster_psize",
                                    "max_cluster_vsize", 
                                    "max_cluster_mass"])

    if per_network: 
        null_df["cluster_network"] = np.tile(NETWORKS, N)

    pool = multiprocessing.Pool(nthreads)

    cluster_stats = pool.starmap(run_fc_permutation, zip(
        [clinical_df] * N, 
        [parcel_df] * N, 
        [moderator_df] * N, 
        [outcomes] * N, 
        [covariates] * N, 
        [moderator] * N, 
        [W_adj] * N, 
        [parcel_sizes] * N, 
        np.repeat(run_id, N),
        perm_ids, 
        [effect] * N, 
        [per_network] * N, 
        [maintain_direction] * N, 
        [shuffle_on] * N, 
        [include_bl] * N, 
        [F_stat] * N, 
        [alpha] * N
    ))
    
    cluster_stats = np.vstack(cluster_stats)

    null_df.iloc[:, 7:] = cluster_stats 

    return null_df

def analyze_permutation(results_df, effect, outcomes, W_adj, parcel_sizes, alpha=0.05, 
                        F_stat=False, per_network=False, maintain_direction=False): 

    if type(results_df) is str: 
        results_df = pd.read_excel(results_df)

    cluster_df = get_clusters(results_df, effect, W_adj, parcel_sizes, outcomes, 
                              alpha=alpha, F_stat=F_stat, 
                              maintain_direction=maintain_direction)
    
    cluster_stats = get_max_clusters(cluster_df, outcomes, per_network)

    return cluster_stats 

def analyze_fc_permutation(run_id, perm_id, permute_dir, effect, moderator, 
                        W_adj, parcel_sizes, alpha=0.05, F_stat=False,
                        per_network=False, maintain_direction=False, 
                        shuffle_on='session'): 

    permute_results = permute_dir.glob(f"Run-{run_id}_Permutation-{perm_id}"
                                f"_*_Moderator-{moderator}_Shuffle-{shuffle_on}_*.xlsx")
    permute_results = [r for r in permute_results if "Clusters" not in str(r)]

    if len(permute_results) > 1:
        sys.stderr.write(f"More than one hit for moderator {moderator}, "
            f"run {run_id}, permutation {perm_id}, shuffled on {shuffle_on}!\n")
    elif len(permute_results) == 0: 
        sys.stderr.write(f"No results file found for moderator {moderator}, "
            f"run {run_id}, permutation {perm_id}, shuffled on {shuffle_on}!\n")
        return cluster_stats.append([np.nan, np.nan, np.nan])
    
    results_fp = permute_results[0]
    print(f"Analyzing results in: {results_fp.name}...")
    results_df = pd.read_excel(permute_results[0])

    cluster_df = get_clusters(results_df, effect, W_adj, parcel_sizes, ["parcel_fc"], 
                            alpha=alpha, F_stat=F_stat, 
                            maintain_direction=maintain_direction)
    
    cluster_stats = get_max_clusters(cluster_df, ["parcel_fc"], per_network)

    return cluster_stats 

""" Code to retrospectively generate null distribution from permutated model fits.
i.e. useful to analyze different effects, significance thresholds post hoc. """
def analyze_permutations(run_id, effect, outcomes, W_adj, parcel_sizes, alpha = 0.05, 
                         F_stat=False, per_network=False, maintain_direction=False, 
                         permute_dir="/scratch/users/rdivya/tms/tmp/permute",
                         nthreads=1):

    permute_dir = Path(permute_dir)
    permute_results = permute_dir.glob(f"Run-{run_id}_Permutation-*_*TreatmentModel_2025*.xlsx") 
    permute_results = [str(r) for r in permute_results]

    N = len(permute_results)
    print("N = ", N)

    n_networks = 1
    if per_network: 
        n_networks = len(NETWORKS)

    null_df = pd.DataFrame(data = {"outcome": np.tile(np.repeat(outcomes,
                                                                 n_networks), 
                                                      N), 
                                "permutation" : np.repeat(np.arange(N), 
                                                len(outcomes) * n_networks), 
                                "cluster_network" : "n/a",
                                "effect" : effect, 
                                "alpha" : alpha, 
                                "F-statistic" : F_stat}, 
                        columns = ["outcome", "permutation", "effect", "alpha", 
                                   "F-statistic", "cluster_network", 
                                    "max_cluster_psize",
                                    "max_cluster_vsize", 
                                    "max_cluster_mass"])
    
    if per_network: 
        null_df["cluster_network"] = np.tile(NETWORKS, N * len(outcomes))

    pool = multiprocessing.Pool(nthreads)

    cluster_stats = pool.starmap(analyze_permutation, zip(
        permute_results, 
        [effect] * N, 
        [outcomes] * N, 
        [W_adj] * N,
        [parcel_sizes] * N,  
        [alpha] * N, 
        [F_stat] * N, 
        [per_network] * N, 
        [maintain_direction] * N
    ))

    cluster_stats = np.vstack(cluster_stats)

    null_df.loc[:, 6:] = cluster_stats 

    return null_df

def analyze_fc_permutations(run_id, effect, moderator, W_adj, parcel_sizes, 
                            alpha=0.05, F_stat=False, per_network=False, 
                            maintain_direction=False, shuffle_on='session',
                            permute_dir="/scratch/users/rdivya/tms/tmp/permute",
                            nthreads=1): 
    
    permute_dir = Path(permute_dir)
    permute_results = permute_dir.glob(f"Run-{run_id}_Permutation-*"
                    f"_*_Moderator-{moderator}_Shuffle-{shuffle_on}_*.xlsx")
    permute_results = [str(r) for r in permute_results if "Clusters" not in str(r)]
    perm_ids = [r.split("Permutation-")[1].split("_")[0] for r in permute_results]

    N = len(perm_ids)
    print("N = ", N)

    n_networks = 1
    if per_network: 
        n_networks = len(NETWORKS)

    null_df = pd.DataFrame(data = {"outcome": "parcel_fc",
                                "moderator": moderator, 
                                "permutation" : np.repeat(perm_ids, n_networks), 
                                "cluster_network" : "n/a",
                                "effect" : effect, 
                                "alpha" : alpha, 
                                "F-statistic" : F_stat}, 
                        columns = ["outcome", "moderator", "permutation", 
                                   "effect", "alpha", "F-statistic", 
                                   "cluster_network", 
                                    "max_cluster_psize",
                                    "max_cluster_vsize", 
                                    "max_cluster_mass"])
    
    if per_network: 
        null_df["cluster_network"] = np.tile(NETWORKS, N)

    pool = multiprocessing.Pool(nthreads)

    cluster_stats = pool.starmap(analyze_fc_permutation, zip(
        [run_id] * N, 
        perm_ids,
        [permute_dir] * N, 
        [effect] * N, 
        [moderator] * N, 
        [W_adj] * N, 
        [parcel_sizes] * N, 
        [alpha] * N,
        [F_stat] * N, 
        [per_network] * N, 
        [maintain_direction] * N, 
        [shuffle_on] * N 
    ))
    
    cluster_stats = np.vstack(cluster_stats)
    null_df.iloc[:, 7:] = cluster_stats 

    return null_df
    
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
    parser.add_argument("--outcomes", nargs="+", default=[],
                         help = "Names of variables in clinical_df to model as outcomes")
    parser.add_argument("--covariates", nargs="*", default=[],
                        help = "Names of variables in clinical_df to model as covariates")
    parser.add_argument("--moderator", help = "Names of variable in <moderator_df> to use as moderator")
    
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
    parser.add_argument('--per_network', action="store_true", help="Classify clusters by network.")
    parser.add_argument("--maintain_direction", action='store_true', 
                        help="Whether the direction of the effect should be kept " \
                        "consistent within clusters formed")
    
    # modes of running this program 
    parser.add_argument("--simulate_null", action='store_true', 
                        help="Simulate null distribution via permutations")
    parser.add_argument("--cluster_effects", action='store_true', 
        help="Analyze experimental results to identify clusters of significant results")
    parser.add_argument("--analyze_permutations", action='store_true', 
                        help="Quantify null effects from previously run permutations")
    parser.add_argument("--get_pvalues", action='store_true', 
        help="Cross reference experimentally observed clusters and simulated null values to assign p-values")

    # optional data frames to input for analysis
    parser.add_argument("--results_df", help="Data frame containing model results to form clusters")
    parser.add_argument("--cluster_df", help="Data frame describing clusters of significant parcels")
    parser.add_argument('--null_df', help="Reference distribution to quantify p-values")
    parser.add_argument("--moderator_df", help="Data frame containing potential moderators of network change.")
    
    # arguments for null simulations
    parser.add_argument("--model_fc", action='store_true', 
                        help="Model FC change. By default, program models symptom change.")    
    parser.add_argument('--shuffle_on', type=str, default='session')
    parser.add_argument('--include_bl', action='store_true')
    parser.add_argument('--lm', action='store_true', 
        help=" Fit a standard linear regression to permuted data at one timepoint only")
    parser.add_argument('--session', type=int, choices=[1,2], default=2)
    parser.add_argument("--permute_dir", type=str, default="/scratch/users/rdivya/tms/tmp/permute", 
                         help="Where results of permutations should be stored")
    
    # options for effects testing 
    parser.add_argument('--alpha', default=0.05, type=float, help="Significance threshold")
    parser.add_argument('--effect', default='parcelEP', help="Main effect of interest")
    parser.add_argument('--F_stat', action='store_true', help="Whether to quantify effect using F-statistic")
    parser.add_argument('--one_sided', action='store_true', help="Whether to conduct a two-sided hypothesis test")

    # options for multi-threading
    parser.add_argument("--run_id", default="Unknown", help="Label describing simulation")
    parser.add_argument("--N", type=int, default=1000, help="Number of permutations to run to simulate null distribution.")
    parser.add_argument("--N_start", type=int, default=0, 
        help="Starting index for permutation number. Set above 0 to concatenate results to a previous run." )
    parser.add_argument("--nthreads", type=int, default=1, help="Number of threads available for multithreading.")

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

    if args.cluster_effects:
        results_df = pd.read_excel(args.results_df) 
        cluster_df = get_clusters(results_df, args.effect, W_adj, psize, args.outcomes, 
                                  F_stat=args.F_stat, alpha=args.alpha, 
                                  maintain_direction=args.maintain_direction)
        
        if args.get_pvalues: 
            null_df = pd.read_excel(args.null_df)
            cluster_df = get_cluster_pvalues(cluster_df, null_df, args.outcomes, 
                                            per_network = args.per_network, 
                                            one_sided=args.one_sided)

        cluster_df.to_excel(args.out_fp, index=False)
    
    # create null distribution using test simulations  
    elif args.simulate_null:
        if args.model_fc:
            if args.moderator_df: 
                if "csv" in args.moderator_df:
                    moderator_df = pd.read_csv(args.moderator_df)
                else:
                    moderator_df = pd.read_excel(args.moderator_df)
            else: 
                moderator_df = None 

            null_df = create_null_fc_distribution(args.N, clinical_df, parcel_df, 
                                           args.outcomes, args.covariates, 
                                           W_adj, psize, args.effect,
                                           moderator_df = moderator_df, 
                                           moderator = args.moderator,
                                           include_bl = args.include_bl,
                                           nthreads=args.nthreads, run_id=args.run_id, 
                                           perm_ids=np.arange(args.N_start, 
                                                        args.N_start + args.N),
                                           F_stat=args.F_stat, alpha=args.alpha,
                                           per_network=args.per_network, 
                                           maintain_direction=args.maintain_direction, 
                                           shuffle_on=args.shuffle_on)
        else:
            null_df = create_null_distribution(args.N, clinical_df, parcel_df, 
                                            args.outcomes, args.covariates, 
                                            W_adj, psize, effect=args.effect, 
                                            nthreads=args.nthreads, run_id=args.run_id, 
                                            perm_ids=np.arange(args.N_start, 
                                                            args.N_start + args.N),
                                            F_stat=args.F_stat, alpha=args.alpha,
                                            per_network=args.per_network, 
                                            maintain_direction=args.maintain_direction)
        null_df.to_excel(args.out_fp, index=False)            

    elif args.analyze_permutations:

        if args.model_fc:
            null_df = analyze_fc_permutations(args.run_id, args.effect, args.moderator,
                                     W_adj, psize, alpha=args.alpha, shuffle_on=args.shuffle_on,
                                     F_stat=args.F_stat, per_network=args.per_network, 
                                     permute_dir=args.permute_dir,
                                     nthreads=args.nthreads)
        else:
            null_df = analyze_permutations(args.run_id, args.effect, args.outcomes, W_adj, psize,  
                                        alpha=args.alpha, F_stat=args.F_stat, nthreads=args.nthreads,
                                        per_network=args.per_network, 
                                        permute_dir=args.permute_dir, 
                                        maintain_direction=args.maintain_direction)
            
        null_df.to_excel(args.out_fp, index=False)

    elif args.get_pvalues:
        cluster_df = pd.read_excel(args.cluster_df)
        null_df = pd.read_excel(args.null_df)
        cluster_df = get_cluster_pvalues(cluster_df, null_df, args.outcomes, 
                                         per_network = args.per_network, 
                                         one_sided=args.one_sided)

        cluster_df.to_excel(args.out_fp, index=False)