import argparse
from datetime import datetime
import os
import multiprocessing
import numpy as np
import pandas as pd
import sys

import utils
import model_effects as model 
import cluster_effects as cluster 

def split_sample(clinical_df, label_df=None):
    if label_df is None: 
        subjects = np.unique(clinical_df["subject"])

        n = len(subjects)
        labels = np.zeros(n)
        labels[:int(n/2)] = 1

        rng = np.random.default_rng()
        rng.shuffle(labels)

        label_df = pd.DataFrame({"subject" : subjects, 
                                 "label" : labels})

    unique_labels = np.unique(label_df["label"])

    label1 = unique_labels[0]
    label2 = unique_labels[1]

    split1_df = pd.merge(clinical_df, 
                      label_df.loc[label_df["label"] == label1], 
                      on = "subject")

    #split1_df.to_excel("debug/split1.xlsx", index=False)

    split2_df = pd.merge(clinical_df, 
                      label_df.loc[label_df["label"] == label2], 
                      on = "subject") 

    #split2_df.to_excel("debug/split2.xlsx", index=False)

    return split1_df, split2_df

def run_split_tx_model(clinical_split_df, parcel_df, outcomes, covariates, 
                    W_adj, parcel_size, null_df, moderator_df=None, moderators=[],
                    predictor="parcel_fc", effect="interaction_session2_parcel", 
                    covary_bl=False, alpha=0.05, model_fc=False): 

    # run lme models on each split of the sample

    if model_fc: 
        results = model.run_lme_fc_tx_model_2(parcel_df, 
                            clinical_split_df, 
                            moderator_df=moderator_df,
                            covariates=covariates, 
                            moderators=moderators,
                            covary_bl=covary_bl)

    else:
        results = model.run_lme_tx_model_2(parcel_df, 
                        clinical_split_df, 
                        outcomes, 
                        predictor=predictor, 
                        covariates=covariates, 
                        covary_bl=covary_bl)

    # cluster effects in  split
    print(effect)
    clusters = cluster.get_clusters(results, effect, W_adj, parcel_size, 
                                    outcomes, alpha=alpha, verbose=True)

    # assign p values based on empirical null distribution
    # (previously calculated based on full sample)
    #clusters = cluster.get_cluster_pvalues(clusters, null_df, 
    #                                        outcomes, 
    #                                        per_network=True)
    
    # only retain clusters that are significant at alpha = 0.05
    #clusters = clusters.loc[clusters["p_perm_mass"] < alpha]

    return results, clusters 

def get_cluster_map(clusters, nparcels=400): 
    parcel_ids = np.arange(nparcels) + 1 # 1-indexing to match Schaefer atlas 

    # initialize cluster map by assigning every parcel into default cluster #0 
    cluster_map = dict.fromkeys(parcel_ids, 0)

    # for each cluster, re-assign relevant parcels into said cluster
    for cluster_num, parcels in (clusters[["cluster_num", "parcel_ids"]]): 
        for parcel in parcels:
            cluster_map[parcel] = cluster_num 

    return cluster_map 

def compare_clusters(clusters1, clusters2, outcomes):

    columns=["outcome", "network_1", "parcels_1", "n1", 
             "network_2", "parcels_2", "n2",
            "dice_coef", "jac_idx"]

    rows = []

    for outcome in outcomes: 
        outcome_clusters1 = clusters1.loc[clusters1["outcome"] == outcome]
        outcome_clusters2 = clusters2.loc[clusters2["outcome"] == outcome]

        outcome_clusters1 = outcome_clusters1.reset_index(drop=True)
        outcome_clusters2 = outcome_clusters2.reset_index(drop=True)

        for idx1, cluster1_row in outcome_clusters1.iterrows(): 
            network1 = cluster1_row["cluster_network"]
            parcels1 = cluster1_row["parcel_ids"]

            for idx2, cluster2_row in outcome_clusters2.iterrows():
                network2 = cluster2_row["cluster_network"]
                parcels2 = cluster2_row["parcel_ids"]

                sep = ", "
                new_row = {"outcome" : outcome, 
                                      "network_1" : network1, 
                                      "parcels_1" : sep.join(parcels1.astype(str)), 
                                      "n1" : len(parcels1),
                                      "network_2" : network2, 
                                      "parcels_2" : sep.join(parcels2.astype(str)),
                                      "n2" : len(parcels2)}

                intersection = np.intersect1d(parcels1, parcels2, assume_unique=True)
                union = np.union1d(parcels1, parcels2)

                new_row["jac_idx"] = len(intersection)/len(union)
                new_row["dice_coef"] = 2*len(intersection)/(len(parcels1) + len(parcels2))

                rows.append(new_row)    

    overlap_df = pd.DataFrame(data=rows, columns=columns)

    return overlap_df

def get_max_overlap(overlap_df, outcomes,
                    control_networks=["somatomotor", "control", "salience", "dorsattn"], 
                    default_networks=["limbic", "default"]):

        in_control = lambda x: (x in control_networks)
        in_default = lambda x: (x in default_networks)

        max_overlap_df = pd.DataFrame({"outcome" : outcomes}, 
                                      columns = ["outcome", 
                                                 "max_overlap_control", 
                                                 "max_overlap_default"])

        for idx, row in max_overlap_df.iterrows():
            outcome = row["outcome"]
            outcome_mask = overlap_df["outcome"] == outcome
            outcome_df = overlap_df.loc[outcome_mask]

            control_mask = np.logical_and(outcome_df["network_1"].map(in_control), 
                                          outcome_df["network_1"].map(in_control))
            default_mask = np.logical_and(outcome_df["network_1"].map(in_default), 
                                        outcome_df["network_1"].map(in_default))

            control_pairs = outcome_df.loc[control_mask, "dice_coef"]
            default_pairs = outcome_df.loc[default_mask, "dice_coef"]

            if len(control_pairs) > 0:
                row["max_overlap_control"] = np.nanmax(outcome_df.loc[control_mask, "dice_coef"])

            if len(default_pairs) > 0:
                row["max_overlap_default"] = np.nanmax(outcome_df.loc[default_mask, "dice_coef"])

            max_overlap_df.iloc[idx] = row

        return max_overlap_df

def compare_parcel_effects(results1, results2, outcomes, effect):
    correlations = []
    statistic = "t_" + effect

    for outcome in outcomes: 
        effect1 = results1.loc[results1["outcome"] == outcome, statistic]
        effect2 = results2.loc[results2["outcome"] == outcome, statistic]

        effect1 = effect1.to_numpy()
        effect2 = effect2.to_numpy()

        nan = np.logical_or(np.isnan(effect1),
                            np.isnan(effect2))
        not_nan = np.logical_not(nan)

        corr = np.corrcoef(effect1[not_nan], 
                           effect2[not_nan])[0,1]

        correlations.append(corr)

    correlation_df = pd.DataFrame(data = {"outcome" : outcomes, 
                                          "correlation" : correlations})

    return correlation_df

def run_split_sample_analysis(clinical_df, parcel_df, null_df,
                              outcomes, covariates, predictor, 
                              W_adj, parcel_size, 
                              label_df=None,
                              moderator_df=None, 
                              effect="interaction_session2_parcel", 
                              tx_model=True,
                              covary_bl=True, 
                              alpha=0.05, 
                              max_overlap=True):

    split1_df, split2_df = split_sample(clinical_df, label_df=label_df)

    if tx_model:
        results1, clusters1 = run_split_tx_model(split1_df, parcel_df, outcomes, 
                                        covariates, W_adj, parcel_size, null_df, 
                                        predictor, effect, covary_bl, alpha)
        #results1.to_excel("debug/results1.xlsx", index=False)
        
        results2, clusters2 = run_split_tx_model(split2_df, parcel_df, outcomes, 
                                        covariates, W_adj, parcel_size, null_df, 
                                        predictor, effect, covary_bl, alpha)
        #results2.to_excel("debug/results2.xlsx", index=False)

    overlap_df = compare_clusters(clusters1, clusters2, outcomes)
    correlation_df = compare_parcel_effects(results1, results2, outcomes, effect)

    if max_overlap:
        max_overlap_df = get_max_overlap(overlap_df, outcomes)
        return max_overlap_df, correlation_df
    
    return overlap_df, correlation_df

def run_split_half_reliability(N, clinical_df, parcel_df, null_df,
                              outcomes, covariates, predictor, 
                              W_adj, parcel_size, 
                              effect="interaction_session2_parcel", 
                              covary_bl=True, 
                              alpha=0.05):
    
    reliability_df = pd.DataFrame(data = {"trial" : np.repeat(np.arange(N), 
                                                              len(outcomes)),
                                        "outcome" : np.tile(outcomes, N)}, 
                                columns=["trial", "outcome", "max_overlap_control", 
                                         "max_overlap_default", "t_stat_corr"])

    for i in range(N):
        overlap_df, correlation_df = run_split_sample_analysis(clinical_df, parcel_df, null_df, 
                                                           outcomes, covariates, predictor, 
                                                           W_adj, parcel_size, effect=effect, 
                                                           covary_bl=covary_bl, alpha=alpha, 
                                                           max_overlap=True)

        row_mask = reliability_df["trial"] == i
        reliability_df.loc[row_mask, "t_stat_corr"] = correlation_df["correlation"]

        reliability_df.loc[row_mask, ["max_overlap_control", 
                                      "max_overlap_default"]] = overlap_df[["max_overlap_control", 
                                                                            "max_overlap_default"]]
        
    return reliability_df

def run_loo(subject, clinical_df, parcel_df, null_df, outcomes, covariates, 
            W_adj, parcel_size, orig_results, orig_clusters, 
            predictor="parcel_fc", covary_bl=True, 
            effect="interaction_session2_parcel", alpha=0.05, 
            run_id="unknown",
            moderator_df=None, moderators=[], model_fc=False,
            tmp_dir="/scratch/users/rdivya/tms/tmp/loocv"): 
    
    loo_clinical_df = clinical_df.loc[clinical_df["subject"] != subject]
    
    loo_results, loo_clusters = run_split_tx_model(loo_clinical_df, parcel_df, 
                                        outcomes, covariates, W_adj, parcel_size, 
                                        null_df, predictor=predictor, effect=effect, 
                                        covary_bl=covary_bl, alpha=alpha, 
                                        moderator_df=moderator_df, 
                                        moderators=moderators,
                                        model_fc=model_fc)

    print(loo_clusters)

    ### store results of permutation to tmp directory ### 
    os.makedirs(tmp_dir, exist_ok=True)

    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H:%M:%S")
    
    results_fn = f"Run-{run_id}_LeaveOut-{subject}_TxModel_{date}_{time}.xlsx" 
    results_fp = f"{tmp_dir}/{results_fn}"
    loo_results.to_excel(results_fp, index=False)

    cluster_fn = f"Run-{run_id}_LeaveOut-{subject}_TxModel_Clusters_{date}_{time}.xlsx" 
    cluster_fp = f"{tmp_dir}/{cluster_fn}"
    loo_clusters.to_excel(cluster_fp, index=False) 

    ### extract LOOCV stats ### 
    overlap_df = compare_clusters(orig_clusters, loo_clusters, outcomes)
    max_overlap_df = get_max_overlap(overlap_df, outcomes)
    overlap_stats = max_overlap_df[["max_overlap_control", "max_overlap_default"]]
    overlap_stats = overlap_stats.to_numpy()

    correlation_df = compare_parcel_effects(orig_results, loo_results, outcomes, effect)
    corr_stats = correlation_df["correlation"].to_numpy()

    loo_stats = np.column_stack((corr_stats, overlap_stats))
    print(loo_stats)
    
    return loo_stats
    
def run_loocv(subject_df, clinical_df, parcel_df, null_df, outcomes, covariates, 
              W_adj, parcel_size, moderator_df=None, moderators=[], model_fc=False, 
              predictor="parcel_fc", covary_bl=True, 
              effect="interaction_session2_parcel", alpha=0.05, 
              run_id="unknown", nthreads=1): 

    clinical_df = pd.merge(clinical_df, 
                           subject_df, 
                           on = "subject")

    orig_results, orig_clusters = run_split_tx_model(clinical_df, parcel_df, 
                                    outcomes, covariates, W_adj, parcel_size, 
                                    null_df, predictor=predictor, effect=effect, 
                                    covary_bl=covary_bl, alpha=alpha, 
                                    moderator_df=moderator_df, 
                                    moderators=moderators,
                                    model_fc=model_fc)

    orig_results.to_excel("debug/orig_results.xlsx", index=False)     

    subjects = np.unique(subject_df["subject"])
    n = len(subjects)

    loocv_df = pd.DataFrame(data = {"left_out_subject" : np.repeat(subjects, 
                                                              len(outcomes)),
                                        "outcome" : np.tile(outcomes, n)}, 
                            columns=["left_out_subject", "outcome", 
                                    "t_stat_corr",
                                    "max_overlap_control", 
                                    "max_overlap_default"])
    stats = ["t_stat_corr", "max_overlap_control", 
            "max_overlap_default"]
    
    pool = multiprocessing.Pool(nthreads)
    
    loo_stats = pool.starmap(run_loo, zip(subjects,
            [clinical_df] * n, 
            [parcel_df] * n,
            [null_df] * n, 
            [outcomes] * n, 
            [covariates] * n, 
            [W_adj] * n, 
            [parcel_size] * n, 
            [orig_results] * n,
            [orig_clusters] * n,
            [predictor] * n, 
            [covary_bl] * n,
            [effect] * n, 
            [alpha] * n, 
            [run_id] * n, 
            [moderator_df] * n, 
            [moderators] * n, 
            [model_fc] * n,
        ))

    loocv_df[stats] = np.vstack(loo_stats)
    return loocv_df
        
def generate_parser():
    parser = argparse.ArgumentParser(prog="cluster_efield_effects", 
            description="Cluster effects from whole-brain models.")
    
    # required args
    parser.add_argument("clinical_df", 
                        help = "File path to workbook containing all outcome variables & covariates.")
    parser.add_argument("parcel_df",
                        help = "Workbook containing values of fixed effects variables for atlas parcels.")
    parser.add_argument('null_df', help="Reference distribution to quantify p-values")
    parser.add_argument("out_fp", help = "Destination file path to store results.")

    # arguments for fitting models 
    parser.add_argument("--model_fc", action='store_true')
    parser.add_argument("--moderator_df")

    # modes of running the program 
    parser.add_argument("--split_sample", action='store_true')
    parser.add_argument("--n_split_sample", action='store_true')
    parser.add_argument("--loocv", action='store_true')

    # arguments for split samples 
    parser.add_argument("--N", default=1, type=int, help="Number of tests to run.")
    parser.add_argument("--label_df")

    # arguments for LOOCV 
    parser.add_argument("--run_id", default="LOOCV")

    # arguments for multithreading
    parser.add_argument("--nthreads", default=1, type=int)

    # arguments for fitting models
    parser.add_argument("--outcomes", nargs="+", default=["qids_tot", "pcl5_tot", 
                                                          "vr36_emo_wellbeing", 
                                                          "vr36_energy", 
                                                          "vr36_role_limit_emo", 
                                                          "vr36_soc_fx"],
                         help = "Names of variables for which to cluster effects")
    parser.add_argument("--covariates", nargs="*", default=["site_scan", "tx_hz"],
                        help = "Names of variables in clinical_df to model as covariates.")
    parser.add_argument("--moderators", nargs="*")
    parser.add_argument("--predictor", default="parcel_fc", type=str, 
                        help="String description of fixed effects variable.")
    parser.add_argument('--covary_bl', action='store_true')

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
    parser.add_argument("--sub_list", help="List of subjects to analyze.")
    parser.add_argument("--add_spatial", action='store_true', 
                        help = "Add spatially adjacent edges to adjacency matrix")
    
    # options for effects testing 
    parser.add_argument('--alpha', default=0.05, type=float, help="Significance threshold")
    parser.add_argument('--effect', default='interaction_session2_parcel', help="Main effect of interest")

    return parser

if __name__ == "__main__":
    parser = generate_parser()
    args = parser.parse_args()

    # load adjacency matrix or create if not found
    sub_list = None
    if args.sub_list is not None and os.path.isfile(args.sub_list):   
        sub_list = pd.read_excel(args.sub_list)

    W_adj = cluster.load_adjacency_matrix(args.fc_adj, args.add_spatial, 
                                  args.fc_threshold, args.percentile, 
                                  task = args.task, 
                                  sub_list=sub_list["subject"].to_numpy(), 
                                  overwrite = args.overwrite)

    # load parcel sizes or calculate if not found
    parcel_size = cluster.load_parcel_size()

    # load data frames
    clinical_df = pd.read_csv(args.clinical_df)
    clinical_df["subject"] = clinical_df["record_id"].map(utils.redcap_to_bids)

    parcel_df = pd.read_excel(args.parcel_df)

    null_df = pd.read_excel(args.null_df)

    moderator_df=None
    if args.moderator_df is not None: 
        moderator_df = pd.read_csv(args.moderator_df)

    label_df=None
    if args.label_df is not None:
        label_df = pd.read_excel(args.label_df)
        label_df["subject"] = label_df["record_id"].map(utils.redcap_to_bids)

    # run generalizability analysis 
    if args.split_sample:
        overlap_df, correlation_df = run_split_sample_analysis(clinical_df, 
                                        parcel_df, 
                                        null_df, 
                                        args.outcomes, 
                                        args.covariates, 
                                        args.predictor, 
                                        W_adj, 
                                        parcel_size, 
                                        label_df=label_df, 
                                        moderator_df=None, 
                                        effect=args.effect, 
                                        covary_bl=args.covary_bl, 
                                        alpha=args.alpha)

        fp_base = args.out_fp.strip(".xlsx") 
        overlap_fp = fp_base + "_overlap.xlsx"
        correlation_fp = fp_base + "_correlation.xlsx"

        overlap_df.to_excel(overlap_fp, index=False)
        correlation_df.to_excel(correlation_fp, index=False)

    elif args.n_split_sample:
        reliability_df = run_split_half_reliability(args.N, clinical_df, parcel_df, null_df,
                              args.outcomes, args.covariates, args.predictor, 
                              W_adj, parcel_size, 
                              effect="interaction_session2_parcel", 
                              covary_bl=True, 
                              alpha=0.05)

        reliability_df.to_excel(args.out_fp, index=False)

    elif args.loocv:
        loocv_df = run_loocv(sub_list, clinical_df, parcel_df, null_df, 
                             args.outcomes, args.covariates, 
                             W_adj, parcel_size, 
                             moderator_df=moderator_df, 
                             moderators=args.moderators,
                             model_fc = args.model_fc,
                             predictor=args.predictor, 
                             covary_bl=args.covary_bl, 
                             effect=args.effect, 
                             alpha=args.alpha, 
                             run_id=args.run_id, 
                             nthreads=args.nthreads)

        loocv_df.to_excel(args.out_fp, index=False)