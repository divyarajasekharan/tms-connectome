import argparse
from datetime import datetime
import multiprocessing
import numpy as np 
import os
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

import sys
import model_effects as model 
import cluster_effects as cluster

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
                    alpha = 0.05, covary_bl=False): 

    tmp_dir = "/scratch/users/rdivya/tms/tmp/permute"

    n_networks = 1
    if per_network: 
        n_networks = len(NETWORKS)

    ### generate permutation model ###    
    parcel_df = permute_data(parcel_df, on = "subject")

    # fit models to permuted data 
    results_df = model.run_lme_tx_model_2(parcel_df, clinical_df, outcomes, 
                                    covariates=covariates, 
                                    covary_bl=covary_bl)
    
    ### store results of permutation to tmp directory ### 
    os.makedirs(tmp_dir, exist_ok=True)

    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H:%M:%S")
    
    results_fn = f"Run-{run_id}_Permutation-{perm_id}_Model-ClinicalChange_{date}_{time}.xlsx" 
    results_fp = f"{tmp_dir}/{results_fn}"
    results_df.to_excel(results_fp, index=False)

    ### for each clinical outcome model, identify clusters of significant parcels ###
    try:
        cluster_df = cluster.get_clusters(results_df, effect, W_adj, parcel_sizes, outcomes, 
                                F_stat=F_stat, alpha=alpha, 
                                maintain_direction=maintain_direction)  

        # store cluster results to tmp directory
        cluster_fn = f"Run-{run_id}_Permutation-{perm_id}_Model-ClinicalChange_Clusters_{date}_{time}.xlsx" 
        cluster_fp = f"{tmp_dir}/{cluster_fn}"
        cluster_df.to_excel(cluster_fp, index=False)

        # extract maximal cluster stats per outcome ### 
        cluster_stats = cluster.get_max_clusters(cluster_df, outcomes, 
                                        per_network=per_network)

    except Exception as e:
        sys.stderr.write(e)
        return [[np.nan, np.nan, np.nan]] * len(outcomes) * n_networks

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
    results_df = model.run_lme_fc_tx_model_2(parcel_df, clinical_df, moderator_df, 
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
    
    cluster_df = cluster.get_clusters(results_df, effect, W_adj, parcel_sizes, outcomes, 
                            F_stat=F_stat, alpha=alpha, 
                            maintain_direction=maintain_direction)  

    ### extract maximal cluster stats per outcome ### 
    cluster_stats = cluster.get_max_clusters(cluster_df, outcomes, 
                                              per_network=per_network)

    return cluster_stats

def create_null_distribution(N, clinical_df, parcel_df, outcomes, covariates,
                             W_adj, parcel_sizes, effect='parcelEP', 
                             covary_bl=False, run_id="Unknown", perm_ids=[], 
                             F_stat=False, alpha=0.05,per_network=False, 
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
        [alpha] * N, 
        [covary_bl] * N
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

    n_networks=1 
    if per_network:
        n_networks=len(NETWORKS)
    if type(results_df) is str: 
        try:
            results_df = pd.read_excel(results_df)
        except Exception as e: 
            print("Error reading file at ", results_df)
            print(e)
            os.remove(results_df)
            return [[np.nan, np.nan, np.nan]] * len(outcomes) * n_networks

    try:    
        cluster_df = cluster.get_clusters(results_df, effect, W_adj, parcel_sizes, outcomes, 
                                alpha=alpha, F_stat=F_stat, 
                                maintain_direction=maintain_direction)
        
        cluster_stats = cluster.get_max_clusters(cluster_df, outcomes, per_network)
    except Exception as e:
        print(e)
        return [[np.nan, np.nan, np.nan]] * len(outcomes) * n_networks

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

    cluster_df = cluster.get_clusters(results_df, effect, W_adj, parcel_sizes, ["parcel_fc"], 
                            alpha=alpha, F_stat=F_stat, 
                            maintain_direction=maintain_direction)
    
    cluster_stats = cluster.get_max_clusters(cluster_df, ["parcel_fc"], per_network)

    return cluster_stats 

""" Code to retrospectively generate null distribution from permutated model fits.
i.e. useful to analyze different effects, significance thresholds post hoc. """
def analyze_permutations(run_id, effect, outcomes, W_adj, parcel_sizes, alpha = 0.05, 
                         F_stat=False, per_network=False, maintain_direction=False, 
                         permute_dir="/scratch/users/rdivya/tms/tmp/permute",
                         nthreads=1):

    permute_dir = Path(permute_dir)
    permute_results = permute_dir.glob(f"Run-{run_id}_Permutation-*.xlsx") 
    permute_results = [str(r) for r in permute_results if "Clusters" not in str(r)]

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
    parser.add_argument("--outcomes", nargs="+", default=["qids_tot", "pcl5_tot", 
                                                          "vr36_emo_wellbeing", 
                                                          "vr36_energy", 
                                                          "vr36_role_limit_emo", 
                                                          "vr36_soc_fx"],
                         help = "Names of variables in clinical_df to model as outcomes")
    parser.add_argument("--covariates", nargs="*", default=["site_scan", "tx_hz"],
                        help = "Names of variables in clinical_df to model as covariates")
    parser.add_argument("--moderator", help = "Names of variable in <moderator_df> to use as moderator")
    parser.add_argument("--covary_bl", action='store_true', help="Whether to inclue baseline outcome value as covariate")

    # options for filtering subjects
    parser.add_argument("--filter_ptsd", type=int, choices=[0,1], 
                        help="Only retain subjects with PTSD (0) or without PTSD (1).")
        
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
    parser.add_argument("--analyze_permutations", action='store_true', 
                        help="Quantify null effects from previously run permutations")
    
    # optional data frames to input for analysis
    parser.add_argument("--moderator_df", help="Data frame containing potential moderators of network change.")
    
    # arguments for null simulations
    parser.add_argument("--model_fc", action='store_true', 
                        help="Model FC change. By default, program models symptom change.")    
    parser.add_argument('--shuffle_on', type=str, default='session')
    parser.add_argument("--permute_dir", type=str, default="/scratch/users/rdivya/tms/tmp/permute", 
                         help="Where results of permutations should be stored")
    
    # options for effects testing 
    parser.add_argument('--alpha', default=0.05, type=float, help="Significance threshold")
    parser.add_argument('--effect', default='interaction_session2_parcel', help="Main effect of interest")
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
    clinical_df["subject"] = clinical_df["record_id"].map(cluster.redcap_to_bids)

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

    W_adj = cluster.load_adjacency_matrix(args.fc_adj, args.add_spatial, 
                                  args.fc_threshold, args.percentile, 
                                  task = args.task, sub_list=sub_list, 
                                  overwrite = args.overwrite)
    
    # load parcel sizes or calculate if not found
    psize_fp = "tmp/schaefer400_parcel_sizes.csv" 

    if os.path.isfile(psize_fp):
        psize = np.loadtxt(psize_fp, delimiter=',', dtype=int)
    else:
        psize = cluster.get_parcel_size()
        np.savetxt(psize_fp, psize, fmt='%d', delimiter=',')

    # filter subjects if needed 
    if args.filter_ptsd is not None:
        no_ptsd = args.filter_ptsd 

        clinical_bl = clinical_df.loc[clinical_df["session"] == 0]
        ptsd_bl = np.array(clinical_bl["pcl5_tot"] >= 33, dtype=np.bool)
        no_ptsd_bl = np.logical_not(ptsd_bl)

        if no_ptsd: 
            clinical_df = pd.merge(clinical_bl.loc[ptsd_bl, "subject"],   
                                    clinical_df, 
                                    on = "subject") 
        else: 
            clinical_df = pd.merge(clinical_bl.loc[ptsd_bl, "subject"],   
                                            clinical_df, 
                                            on = "subject") 
        
    # create null distribution using test simulations  
    if args.simulate_null:
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
                                           include_bl = args.covary_bl,
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
                                            maintain_direction=args.maintain_direction, 
                                            covary_bl=args.covary_bl)
        null_df.to_excel(args.out_fp, index=False)            

    if args.analyze_permutations:

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