import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps as cm
import nibabel as nib
import numpy as np 
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multitest
import sys 

# silence excessive warnings from pandas
import warnings
warnings.simplefilter(action='ignore', category=Warning)

TMS_DIR="/oak/stanford/groups/leanew1/SHARED_DATASETS/private/tms/bids/data"
FS_DIR = f"{TMS_DIR}/derivatives/fmriprep-20.2.3/freesurfer"

DATA_DIR="/oak/stanford/groups/leanew1/users/rdivya/tms/data"
ATLAS_DIR=f"{DATA_DIR}/atlases"
SCHAEFER_ATLAS_SURF=f"{ATLAS_DIR}/Schaefer/fsaverage/lh.Schaefer2018_400Parcels_17Networks_order.annot"

SGACC_PARCELS = ["17Networks_LH_DefaultA_PFCm_2", "17Networks_RH_DefaultA_PFCm_1"]
DACC_PARCELS = ["17Networks_LH_ContA_Cingm_1", "17Networks_RH_ContA_Cingm_1"]
DLPFC_PARCELS = ["17Networks_LH_ContA_PFCl_1", "17Networks_LH_ContA_PFCl_2", 
                 "17Networks_LH_ContA_PFCl_3", "17Networks_LH_ContA_PFClv_1",  
                 "17Networks_LH_ContA_PFClv_2", "17Networks_RH_ContA_PFCl_1",
                  "17Networks_RH_ContA_PFCl_2", "17Networks_RH_ContA_PFCl_3",
                  "17Networks_RH_ContA_PFCl_4", "17Networks_RH_ContA_PFCl_5" ]

SITE_MAP = {'1991' : 'PA', 
            '1976' : 'PV', 
            '2000' : 'MN', 
            '1984' : 'WJ', 
            '1992' : 'MY'
}

DEBUG=True

def combine_atlases(atlases, merge_on):

    df = pd.read_excel(atlases[0])

    for atlas in atlases[1:]: 
        atlas = pd.read_excel(atlas)
        df = pd.merge(df, atlas, on = merge_on)
        
    return df

def redcap_to_bids(redcap_id):
    
    site_id, sub_id = redcap_id.split("-")

    return "TMS" + SITE_MAP[site_id] + sub_id.zfill(3)

def get_responders(clinical_df): 
    change_df = compute_tx_change(clinical_df[["subject", "session", "qids_tot"]], 
                                  ["qids_tot"])
    
    responder_df = change_df[change_df.session == 2]
    responder_df["qids_perc_change"] = responder_df["qids_tot_change"] / responder_df["qids_tot_bl"] 
    responder_df["responder"] = responder_df["qids_perc_change"] <= -0.5 

    return responder_df[["subject", "responder"]]

def get_bl_stim_sgacc_fc(parcel_df): 

    sgacc_df = parcel_df[parcel_df.session == 0]
    sgacc_df["bl_stim_sgacc_fc"] = np.nanmean(sgacc_df[SGACC_PARCELS], 
                                              axis=1)
    sgacc_df["on_target"] = sgacc_df["bl_stim_sgacc_fc"] < 0

    return sgacc_df[["subject", "bl_stim_sgacc_fc"]]

def get_bl_stim_dlpfc_fc(parcel_df): 

    sgacc_df = parcel_df[parcel_df.session == 0]
    sgacc_df["bl_stim_dlpfc_fc"] = np.nanmean(sgacc_df[DLPFC_PARCELS], 
                                              axis=1)

    return sgacc_df[["subject", "bl_stim_dlpfc_fc"]]

def get_bl_stim_dacc_fc(parcel_df): 

    sgacc_df = parcel_df[parcel_df.session == 0]
    sgacc_df["bl_stim_dacc_fc"] = np.nanmean(sgacc_df[DACC_PARCELS], 
                                              axis=1)

    return sgacc_df[["subject", "bl_stim_dacc_fc"]]

def add_efield_stat(parcel_df, stat="peak"):
    parcels = get_parcels(parcel_df)

    if stat == "avg":
        parcel_df["efield_avg"] = np.mean(parcel_df[parcels], 
                                        axis=1)
    elif stat == "peak":
        parcel_df["efield_peak"] = np.max(parcel_df[parcels], 
                                        axis=1)

    return parcel_df 

def normalize_predictor(parcel_df, method="zscore"): 
    parcels = get_parcels(parcel_df)

    # z-normalize across subjects
    if "zscore_across": 
        for col in parcel_df.columns:
            if col == "subject": 
                continue
                
            col_mean = np.mean(parcel_df[col])
            col_std = np.std(parcel_df[col])

            parcel_df[col] = (parcel_df[col] - col_mean)/col_std

    for idx, row in parcel_df.iterrows(): 

        if method == "max":
            row_max = np.max(row[parcels])
            row[parcels] /= row_max 
        elif method == "zscore":
            row_mean = np.mean(row[parcels])
            row_sd = np.std(row[parcels])

            row[parcels] = (row[parcels] - row_mean)/row_sd
        elif method == "log":
            row[parcels] = np.log(row[parcels].to_numpy(dtype=np.float))
        
        parcel_df.iloc[idx] = row
    
    return parcel_df
        
def compute_tx_change(data_df, change_vars): 
    ### compute change scores ##
    
    # extract baseline values for change scores
    bl_df = data_df.loc[data_df.session == 0, change_vars + ["subject"]]
    bl_df.columns = [col + "_bl" if col != "subject" else col 
                     for col in bl_df.columns]

    # extract post-baseline sessions & compute changes
    data_df = data_df[data_df.session != 0]

    data_df = pd.merge(data_df, bl_df, on = "subject")

    for var in change_vars: 
        data_df[f"{var}_change"] = (data_df[var] - data_df[f"{var}_bl"])

    # only retain baseline and change values for modified columns
    data_df.drop(columns = change_vars, inplace=True)

    return data_df

def code_factors(data_df, factor_map):
    ml_factors = []
    code_map = {}

    for factor, [ref, base] in factor_map.items(): 
        # determine which factor levels to code as dummy regressors 
        code_levels = list(np.unique(data_df[factor]))

        if ref in code_levels:
            code_levels.remove(ref)

        if len(code_levels) > 1: 
            ml_factors.append(factor)

        dummy_vars = []
        for level in code_levels: 
            dummy_var = base + str(level).upper()
            dummy_vars.append(dummy_var)

            data_df[dummy_var] = data_df[factor].map(lambda x: 1 if x == level else 0)
        
        # link factor variable with coded dummy regressors
        code_map[factor] = dummy_vars

    return data_df, ml_factors, code_map 

def create_factor_map(factors, ref_ses=0): 
    factor_map = {}

    if "session" in factors: 
        factor_map["session"] = [ref_ses, 'session']

    if "tx_hz" in factors: 
        factor_map["tx_hz"] = [10, "hz"]

    if "site_scan" in factors: 
        factor_map["site_scan"] = [1976, "site"]

    if "biotype" in factors: 
        factor_map["biotype"] = ["C-", "biotype"]

    if "cluster" in factors: 
        factor_map["cluster"] = [0, "cluster"]
    
    if "responder" in factors: 
        factor_map["responder"] = [False, "responder"]

    if "on_target" in factors: 
        factor_map["on_target"] = [False, "ontarget"]

    if "seg_low" in factors: 
        factor_map["seg_low"] = [False, "seg_low"]

    if "ptsd" in factors:
        factor_map["ptsd"] = [False, "ptsd"]
        
    if "attention_biotype" in factors: 
        factor_map["attention_biotype"] = ["normal", "attention"]
    
    if "dmn_biotype" in factors: 
        factor_map["dmn_biotype"] = ["normal", "dmn"]

    return factor_map

def get_parcels(parcel_df): 
    # compile a list of all parcels from column names 
    # (excluding subject & session)
    parcels = parcel_df.columns.to_list()

    if "subject" in parcels:
        parcels.remove("subject")
    
    if "session" in parcels:
        parcels.remove("session")

    if "efield_peak" in parcels:
        parcels.remove("efield_peak")

    if "efield_avg" in parcels:
        parcels.remove("efield_avg")

    return parcels

def get_model_formula(predictor, covariates, random_effects): 
    # write out model formula  
    formula = f"outcome ~ {predictor}"

    # add covariates and random effects
    for cov in covariates: 
        formula = formula + f" + {cov}"

    for re in [random_effects]:
        formula = formula + f" + (1 | {re})"
    
    return formula 

def setup_results_df(parcels, predictor, outcomes, covariates, 
                     fixed_effects, random_effects, ml_factors=[],
                     moderators=[]): 
    
    formula = get_model_formula(predictor, covariates, random_effects)
    parcel_id = np.arange(len(parcels)) + 1
    
    results_df = pd.DataFrame(
        data = {"parcel" : np.tile(parcels, len(outcomes)), 
                "parcel_id" : np.tile(parcel_id, len(outcomes)), 
                "outcome" : np.repeat(outcomes, len(parcels)), 
                "covariates" : ", ".join(covariates),
                "moderator" : ",".join(moderators), 
                "predictor" : predictor, 
                "model" : formula}, 
        columns = ["parcel", "parcel_id", "predictor", "outcome", "moderator", 
                    "covariates", "model","n", "df", "reml", "converged", 
                    "model_marg_r2", "model_cond_r2", "parcel_marg_r2", 
                    "parcel_cond_r2", "aic", "bic"])
    
    results_df.reset_index(drop=True, inplace=True)
    
    for stat in ["coef", "se", "t", "p"]:
        for fe in fixed_effects: 
            results_df.insert(len(results_df.columns), f"{stat}_{fe}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add test statistics & inferences for multi-level factors & overall model
    for p in ml_factors + moderators + ["model", "parcel"]: 
        if f"p_{p}" not in results_df.columns:
            results_df.insert(len(results_df.columns), f"F_{p}", np.nan)
            results_df.insert(len(results_df.columns), f"p_{p}", np.nan)
    
    return results_df 

def write_model_fit(row, fit, fixed_effects, random_effects, 
                    model="lmer"): 
    
    for fe in fixed_effects:
        if model=="lmer":
            row[f"coef_{fe}"] = fit.fe_params[fe]
            row[f"se_{fe}"] = fit.bse_fe[fe]

        elif model=="ols":
            row[f"coef_{fe}"] = fit.params[fe]
            row[f"se_{fe}"] = fit.bse[fe]

        row[f"t_{fe}"] = fit.tvalues[fe]
        row[f"p_{fe}"] = fit.pvalues[fe]

    if model=="lmer":
        for i, re in enumerate([random_effects]):
            row[f"var_{re}"] = fit.cov_re.iloc[i,i]
        
        row["converged"] = fit.converged

    row["aic"] = fit.aic
    row["bic"] = fit.bic

    return row 

def compute_r_squared(model_fit, model_df, verbose=False):
    var_resid = model_fit.scale
    var_random_effect = float(model_fit.cov_re.iloc[0])
    var_fixed_effect = model_fit.predict(model_df).var()

    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

    if verbose:
        print("Marginal r-squared: ", marginal_r2)
        print("Conditional r-squared: ", conditional_r2)

    return conditional_r2, marginal_r2

def test_multi_level_factor_effects(ml_factors, code_map, model_fit, reg_id, row): 
    # implement contrasts to test fixed effect of multi-level factors
    for factor in ml_factors: 
        levels = code_map[factor]
        n_vars = len(levels)

        r_matrix = np.zeros(shape = (n_vars, len(model_fit.params)))

        for i, level in enumerate(levels):
            r_matrix[i, reg_id[level]] = 1

        try: 
            f_test = model_fit.f_test(r_matrix)
        except Exception as e: 
            sys.stderr.write(f"F-test failed for variable {factor} with message: {str(e)}.\n")
            return row 
        
        row[f"F_{factor}"] = f_test.fvalue
        row[f"p_{factor}"] = f_test.pvalue
    
    return row 

def run_lme_tx_model_2(parcel_df, clinical_df, outcomes, 
                  predictor = "fc_change", covariates=[], 
                  random_effects="subject", ref_ses=0, 
                  bl_predictor=False, perc_change=False,
                  merge_on = ["subject", "session"], 
                  norm=False, norm_how="zscore", 
                  estimate_r2=False, 
                  reml=False, fdr=False, out_df = None, 
                  verbose=False, covary_bl=False): 

    # normalize predictor data if requested
    if norm:
        parcel_df = normalize_predictor(parcel_df, method=norm_how)

    # add parcel-based statistics if requested
    # TO-DO: does not need to be efield specific 
    if "efield_peak" in covariates:
        parcel_df = add_efield_stat(parcel_df, stat="peak")

    if "efield_avg" in covariates:
        parcel_df = add_efield_stat(parcel_df, stat="avg")

    parcels = get_parcels(parcel_df)

    # only retain variables needed for model

    # assume remaining covariates are stored in parcel_df, or will
    # be computed upon the compute_tx_change operation
    # otherwise, a KeyError will be raised during fitting steps
    cov_tmp = [c for c in covariates if c in clinical_df.columns]
    
    clinical_df = clinical_df[["subject", "session"] + outcomes + cov_tmp]

    # merge predictor and clinical outcome data 
    data_df = pd.merge(clinical_df, 
                       parcel_df, 
                       on = merge_on)

    data_df.to_excel("debug/lme_tx_model_data_orig.xlsx", index=False)

    ### compute change scores and remove rows for baseline session ##
    # which variables should be computed as a delta referenced to baseilne 
    # by default, change in outcomes are modeled
    change_vars = [] + outcomes 
    if not bl_predictor:
        change_vars += parcels 

    data_df = compute_tx_change(data_df, change_vars)

    data_df.to_excel("debug/lme_tx_model_data_change.xlsx", index=False)

    ### code factor levels as dummy variables ###
    predictors = ["session"] + covariates
    
    if ref_ses in [1,2]: 
        predictors += ["parcel"]
        
    factor_map = create_factor_map(predictors, ref_ses)
    data_df, ml_factors, code_map = code_factors(data_df, factor_map)

    # concatenate list of dummy regressors coded for each factor variable
    factor_vars = list(code_map.keys())
    dummy_vars = []
    for vars in code_map.values(): 
        dummy_vars = dummy_vars + vars

    # identify continuous predictors  
    continuous_vars = predictors.copy()
    for x in factor_vars: 
        continuous_vars.remove(x)

    if covary_bl: 
        continuous_vars += ["outcome_bl"] # parcel_bl

    # summarize types of predictor included in model 
    if verbose:
        print("Factor variables: ", factor_vars)
        print("Continuous predictors: ", continuous_vars)
        print("Outcomes: ", outcomes)

    ### specify session x parcel interaction terms ###     
    interaction_vars = []
    ses_vars = code_map["session"] 

    for ses in ses_vars:
        interaction_vars.append("interaction_" + ses + "_parcel")

    ### finally, specify all explicitly coded fixed effect regressors ### 
    fixed_effects = continuous_vars + dummy_vars + interaction_vars 
    base_effects = continuous_vars + dummy_vars 

    if verbose:
        print("Fixed effects: ", fixed_effects)

    num_id = dict(zip(fixed_effects, range(len(fixed_effects)))) 

    ### set up data frame to store results ### 
    
    # hack
    # if an outcome is modeled as a covariate, exclude it
    outcomes = [o for o in outcomes if o + "_change" not in covariates]

    results_df = setup_results_df(parcels, predictor, outcomes, 
                                  covariates, fixed_effects, random_effects, 
                                  ml_factors)
    
    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        outcome = row["outcome"]
        parcel = row["parcel"]

        if verbose:
            print("Outcome: ", outcome)
            print("Parcel: ", parcel)
        
        # standardize parcel predictor name & outcome name for convenience 
        if perc_change:
            data_df["outcome"] = data_df[f"{outcome}_change"]/data_df[f"{outcome}_bl"]
        else: 
            data_df["outcome"] = data_df[f"{outcome}_change"]

        data_df["outcome_bl"] = data_df[f"{outcome}_bl"]

        if bl_predictor:
            data_df["parcel"] = data_df[f"{parcel}"]
        else:
            parcel_bl = parcel + "_bl"

            data_df["parcel_bl"] = data_df[parcel_bl]
            data_df["parcel"] = data_df[f"{parcel}_change"]

        # compute interaction terms
        for int in interaction_vars: 
            ses = int.split("_")[1]
            data_df[int] = data_df[ses] * data_df["parcel"]

        # extract model regressors  
        model_df = data_df[fixed_effects + ["outcome"] + [random_effects]]

        # remove rows with missing model variables
        model_df.replace({np.inf : np.nan}, inplace=True) # remove cases where baseline score was 0
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)
        row["df"] = len(model_df) 
        row["n"] = len(np.unique(model_df["subject"]))

        # construct model 
        model = sm.MixedLM(model_df["outcome"], model_df[fixed_effects], 
                            groups = model_df[random_effects])

        base_model = sm.MixedLM(model_df["outcome"], model_df[base_effects], 
                            groups = model_df[random_effects])
        # fit model
        try:
            model_fit = model.fit(method=['cg'], reml=reml)
        except Exception as e:
            print(f"Fitting failed for {parcel} with error message: {e}")
            
            row["converged"] = False 
            results_df.iloc[idx] = row

            continue
    
        if verbose:
            print(model_fit.summary())

        # write out parameters of fitted model 
        row = write_model_fit(row, model_fit, fixed_effects, random_effects)

        # implement contrasts to test overall model fit
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        try: 
            f_test = model_fit.f_test(r_matrix)
            row["F_model"] = f_test.fvalue
            row["p_model"] = f_test.pvalue
        except Exception as e:
            print(e)

        # implement contrasts to test main effect of parcel fc change 
        r_matrix = np.zeros(shape = (len(interaction_vars), len(model_fit.params)))

        for i, int in enumerate(interaction_vars):
            r_matrix[i, num_id[int]] = 1

        try: 
            f_test = model_fit.f_test(r_matrix)
            row["F_parcel"] = f_test.fvalue
            row["p_parcel"] = f_test.pvalue
        except Exception as e: 
            print(e)

        # implement contrasts to test fixed effect of session
        row = test_multi_level_factor_effects(ml_factors, code_map, model_fit, 
                                              num_id, row)   

        # implement Nakagawa & Schielzeth (2012) method to estimate r-squared
        if estimate_r2:
            r2s = compute_r_squared(model_fit, model_df[fixed_effects], verbose)
            
            row[["model_cond_r2", "model_marg_r2"]] = r2s 

            # fit base model
            try:
                base_model_fit = base_model.fit(method=['cg'], reml=reml)
            except Exception as e:
                print(f"Fitting of base model failed with error message: {e}")
                results_df.iloc[idx] = row
                continue

            r2s_base = compute_r_squared(base_model_fit, model_df[base_effects], verbose=False)

            row[["parcel_cond_r2", "parcel_marg_r2"]] = np.array(r2s) - np.array(r2s_base)

        results_df.iloc[idx] = row

    # FDR correction using Benjamini/Hochberg method
    if fdr: 
        _, p_corr, _, _ = multitest.multipletests(results_df["p_parcel"], method = 'fdr_bh')
        results_df["p_parcel_corr"] = p_corr

    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df.dropna(how='all', axis=1)

def run_lm_tx_model(parcel_df, clinical_df, outcomes, session, 
                  predictor = "parcel_fc", bl_predictor=False,
                  covariates=[], moderator=None,
                  covary_bl=False,
                  merge_on = ["subject", "session"], 
                  fdr=False, out_df = None, verbose=False): 

    parcels = get_parcels(parcel_df)

    if moderator is not None:
        covariates += [moderator]

    # only retain variables needed for model
    vars = ["subject", "session"] + outcomes + covariates
    clinical_df = clinical_df[vars]

    # merge predictor and clinical outcome data 
    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = merge_on)

    data_df.to_excel(f"debug/lm_tx_model_data_orig.xlsx", index=False)

    ### compute change scores and remove rows for baseline session ##
    change_vars = outcomes + []

    if not bl_predictor:
        change_vars += parcels 

    data_df = compute_tx_change(data_df, change_vars)
    data_df = data_df[data_df.session == session]
    data_df.to_excel(f"debug/lm_tx_model_data_change_ses-{session}.xlsx", index=False)

    ### code factor levels as dummy variables ###
    predictors = covariates

    factor_map = create_factor_map(predictors)
    data_df, ml_factors, code_map = code_factors(data_df, factor_map)

    # concatenate list of dummy regressors coded for each factor variable
    factor_vars = list(code_map.keys())
    dummy_vars = []
    for x in code_map.values(): 
        dummy_vars = dummy_vars + x

    # identify continuous predictors  
    continuous_vars = predictors.copy()
    for x in factor_vars: 
        continuous_vars.remove(x)

    # summarize types of predictor included in model 
    print("Factor variables: ", factor_vars)
    print("Continuous predictors: ", continuous_vars)

    ### finally, specify all explicitly coded fixed effect regressors ### 
    fixed_effects = ["intercept", "parcel"] + continuous_vars + dummy_vars 

    if covary_bl: 
        fixed_effects += ["parcel_bl"]
    
    if moderator is not None: 
        interaction_vars = []

        if moderator in factor_vars: 
            print("Moderator is a factor variable!")
            for level in code_map[moderator]:
                var_name = "interaction_parcel_" + level
                interaction_vars.append(var_name)
        else: 
            print("Moderator is a continuous variable!")
            var_name = "interaction_parcel_" + moderator
            interaction_vars.append(var_name)

        fixed_effects += interaction_vars

    print("Fixed effects: ", fixed_effects)

    num_id = dict(zip(fixed_effects, range(len(fixed_effects)))) 

    ### set up data frame to store results ### 

    results_df = setup_results_df(parcels, predictor, outcomes, 
                                  covariates, fixed_effects, [], 
                                  ml_factors)

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        outcome = row["outcome"]
        parcel = row["parcel"]
        parcel_bl = parcel + "_bl"


        if moderator is None:
            formula = f"{outcome}_change ~ 1 + {parcel}_change"
        else: 
            formula = f"{outcome}_change ~ 1 + {moderator} x {parcel}_change"

        for cov in covariates: 
            formula = formula + f" + {cov}"

        row["model"] = formula

        if verbose:
            print("Fitting model: ", formula)

        # standardize predictor and outcome variable names for convenience 
        data_df["outcome"] = data_df[f"{outcome}_change"]

        if bl_predictor:
            data_df["parcel"] = data_df[f"{parcel}"]
        else:
            data_df["parcel"] = data_df[f"{parcel}_change"]
            data_df["parcel_bl"] = data_df[parcel_bl]


        # add model intercept
        data_df["intercept"] = 1

        # add interaction term 
        if moderator is not None:
            if moderator in factor_vars: 
                for level in code_map[moderator]:
                    var_name = "interaction_parcel_" + level
                    data_df[var_name] = data_df["parcel"] * data_df[level]
            else: 
                var_name = "interaction_parcel_" + moderator
            
                data_df[var_name] = data_df["parcel"] * data_df[moderator]

        model_df = data_df[["subject", "outcome"] + fixed_effects]

        # remove rows with missing model variables 
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)
 
        # construct model 
        model = sm.OLS(model_df["outcome"], model_df[fixed_effects])
        
        # fit model
        try:
            model_fit = model.fit()
            row["converged"] = True
        except Exception as e:
            print(f"Model fit failed with error message: {e}")
            
            row["converged"] = False 
            results_df.iloc[idx] = row

            continue
    
        if verbose:
            print(model_fit.summary())

        # write out parameters of fitted model 
        row = write_model_fit(row, model_fit, fixed_effects, None, 
                              model="ols")

        # implement contrasts to test fixed effect of all multi-level factor variables
        row = test_multi_level_factor_effects(ml_factors, code_map, model_fit, 
                                              num_id, row)   

        results_df.iloc[idx] = row

    # FDR correction using Benjamini/Hochberg method
    if fdr: 
        results_df["p_parcel_corr"] = np.nan

        for outcome in outcomes:
            _, p_corr, _, _ = multitest.multipletests(results_df.loc[results_df.outcome == outcome, 
                                                                     "p_parcel"], method = 'fdr_bh')
            results_df.loc[results_df.outcome == outcome, 
                           "p_parcel_corr"] = p_corr

    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def run_lme_fc_tx_model_2(parcel_df, clinical_df, moderator_df=None, 
                        covariates=[], moderators=[],
                        random_effects="subject", reml=False,
                        covary_bl = True, out_df = None):
    
    # get list of parcels in data frame 
    parcels = get_parcels(parcel_df)

    # only retain variables included in model from clinical dataset
    cov_tmp = [c for c in covariates if c in clinical_df.columns]
    clinical_df = clinical_df[["subject", "session"] + cov_tmp]

    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = ["subject", "session"])
    
    # add moderators 
    if moderator_df is not None: 
        data_df = pd.merge(data_df, 
                           moderator_df, 
                           on = ["subject"], 
                           how = "outer")

    # compile variables to be included as predictors in models 
    # moderators added later as part of interaction terms
    predictors = ["session"] + covariates

    ### compute delta values and remove rows for baseline session ##
    data_df = compute_tx_change(data_df, parcels)

    if DEBUG: 
        data_df.to_excel(f"debug/fc_change_data_mod_{moderators}.xlsx", index=False)

    ### code factor levels as dummy variables ###
    # identify which of the modeled variables are factor variables
    factor_map = create_factor_map(predictors + moderators)
    factor_vars = list(factor_map.keys())
    print("Factor variables: ", factor_vars)
   
    # code factor variables using dummy variables 
    data_df, ml_factors, code_map = code_factors(data_df, factor_map)

    # concatenate list of dummy regressors coded for each factor variable
    # EXCLUDE moderator terms, which will be included as interactions
    dummy_vars = []
    for factor, vars in code_map.items(): 
        if factor not in moderators: 
            dummy_vars = dummy_vars + vars

    # identify continuous variables 
    continuous_vars = predictors.copy()
    for x in factor_vars: 
        if x not in moderators: 
            continuous_vars.remove(x)

    print("Continuous predictors: ", continuous_vars)

    ### compute interaction terms ### 
    interaction_vars = []
    ses_vars = code_map["session"] 
    print("Session variables: ", ses_vars)
    print("Moderator: ", moderators)

    for ses in ses_vars: 
        for mod in moderators:
            if mod in factor_vars: 
                print("Moderator is a factor variable!")
                for level in code_map[mod]:
                    var_name = "interaction_" + ses + "_" + level

                    data_df[var_name] = data_df[ses] * data_df[level]

                    interaction_vars.append(var_name)
            else: 
                print("Moderator is a continuous variable!")

                var_name = "interaction_" + ses + "_" + mod

                data_df[var_name] = data_df[ses] * data_df[mod]

                interaction_vars.append(var_name)

    print("Interaction variables: ", interaction_vars)

    ### specify explicitly coded fixed effect regressors ###
    if covary_bl: 
        continuous_vars += ["parcel_bl"]
    
    fixed_effects = continuous_vars + dummy_vars + interaction_vars
    num_id = dict(zip(fixed_effects, range(len(fixed_effects)))) 

    print("Fixed effect regressors: ", fixed_effects)
        
    # write out fixed effect structure   
    predictor = ""
    if len(moderators) > 0:
        predictor = moderators[0] + " x session"

        for moderator in moderators[1:]:        
            predictor += " + " + moderator + " x session"
    else:
        predictor = "session"

    ### set up data frame to store results ### 
    results_df = setup_results_df(parcels, predictor, ["parcel_fc"], covariates,
                                  fixed_effects, random_effects, 
                                  ml_factors, moderators)

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        parcel = row["parcel"] 
        parcel_change = parcel + "_change"
        parcel_bl = parcel + "_bl"
        print("Parcel: ", parcel)

        # standardize variable names 
        data_df["parcel_bl"] = data_df[parcel_bl]
        
        # extract data needed to fit model & remove rows with incomplete data
        model_df = data_df[fixed_effects + [parcel_change] + [random_effects]]
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        n = len(np.unique(model_df["subject"]))
        row["n"] = n
        print(f"n = {n}")

        # fit model using statsmodels 
        model = sm.MixedLM(model_df[parcel_change], model_df[fixed_effects], 
                            groups = model_df[random_effects])
        
        try:
            model_fit = model.fit(method = ['cg'], reml=reml)
        
            print(model_fit.summary())
        except Exception as e: 
            sys.stderr.write(f"Model fit failed for {parcel} with error {str(e)}.\n")
            continue 

        # write out fitted model coefficients 
        row = write_model_fit(row, model_fit, fixed_effects, random_effects)

        # implement contrasts to test fixed effect of multi-level factors
        test_factors = ml_factors.copy()

        try:
            test_factors.remove(moderators)
        except ValueError:
            pass

        row = test_multi_level_factor_effects(test_factors, code_map, model_fit, 
                                              num_id, row)

        # test overall effect of moderator, if included and a multi-level factor 
        # not valid for multiple moderators
        if len(interaction_vars) > 1: 
            n_vars = len(interaction_vars)
            r_matrix = np.zeros(shape = (n_vars, len(model_fit.params)))

            for i, int in enumerate(interaction_vars): 
                r_matrix[i, num_id[int]] = 1

            try: 
                f_test = model_fit.f_test(r_matrix)

                row[f"F_{moderators[0]}"] = f_test.fvalue
                row[f"p_{moderators[0]}"] = f_test.pvalue 
            except Exception as e: 
                sys.stderr.write(f"F-test failed for variable {moderator} with message: {str(e)}.\n")

        # test overall model significance
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        try: 
            f_test = model_fit.f_test(r_matrix)
            row["F_model"] = f_test.fvalue
            row["p_model"] = f_test.pvalue
        except Exception as e: 
            sys.stderr.write(f"F-test failed for full model with message: {str(e)}.\n")
        
        results_df.iloc[idx] = row
    
    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def run_lme_fc_model_2(parcel_df, clinical_df, covariates=[], moderator_df=None, 
                        random_effects="subject", reml=False, out_df = None):
    
    # get list of parcels in data frame 
    parcels = get_parcels(parcel_df)

    # only retain variables included in model from clinical dataset
    clinical_df = clinical_df[["subject", "session"] + covariates]

    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = ["subject", "session"])
    
    # check if model should include a moderator 
    # merge moderator data
    moderators = []
    if moderator_df is not None: 
        moderators = moderator_df.columns.to_list()
        merge_on = []

        for var in ["subject", "session"]: 
            if var in moderators:
                merge_on.append(var)
                moderators.remove(var)

        data_df = pd.merge(data_df, moderator_df, 
                           on = merge_on)

    # compile variables to be included as predictors in models 
    predictors = ["session"] + moderators + covariates

    ### code factor levels as dummy variables ###
    # identify which of the modeled variables are factor variables
    factor_map = create_factor_map(predictors + moderators)
    factor_vars = list(factor_map.keys())
    print("Factor variables: ", factor_vars)
   
    # code factor variables using dummy variables 
    data_df, ml_factors, code_map = code_factors(data_df, factor_map)

    # concatenate list of dummy regressors coded for each factor variable
    # EXCLUDE moderator terms, which will be included as interactions
    dummy_vars = []
    for factor, vars in code_map.items(): 
        dummy_vars = dummy_vars + vars

    # identify continuous variables 
    continuous_vars = predictors.copy()
    for x in factor_vars: 
        if x not in moderators: 
            continuous_vars.remove(x)

    print("Continuous predictors: ", continuous_vars)

    ### compute interaction terms ### 
    interaction_vars = []
    ses_vars = code_map["session"] 
    print("Session variables: ", ses_vars)
    print("Moderators: ", moderators)
    for mod in moderators: 
        for ses in ses_vars: 
            if mod in factor_vars: 
                print("Moderator is a factor variable!")
                for level in code_map[mod]:
                    var_name = "interaction_" + ses + "_" + level

                    data_df[var_name] = data_df[ses] * data_df[level]

                    interaction_vars.append(var_name)
            else: 
                print("Moderator is a continuous variable!")

                var_name = "interaction_" + ses + "_" + mod

                data_df[var_name] = data_df[ses] * data_df[mod]

                interaction_vars.append(var_name)

    print("Interaction variables: ", interaction_vars)

    ### specify explicitly coded fixed effect regressors ###
    continuous_vars += ["intercept"] # explicitly include intercept in model w/ baseline
    fixed_effects = continuous_vars + dummy_vars + interaction_vars
    num_id = dict(zip(fixed_effects, range(len(fixed_effects)))) 

    print("Fixed effect regressors: ", fixed_effects)
        
    # write out fixed effect structure   
    if len(moderators) > 0:
        predictor = moderators[0]

        for mod in moderators[1:]: 
            predictor = predictor + " x " + mod 
        
        predictor = predictor + " x session"
    else:
        predictor = "session"

    ### set up data frame to store results ### 
    results_df = setup_results_df(parcels, predictor, ["parcel_fc"], covariates, 
                                  fixed_effects, random_effects, ml_factors)

    if len(interaction_vars) > 1: 
        mod = moderators[0]

        if mod not in ml_factors:
            results_df.insert(len(results_df.columns), f"F_{mod}_interaction", np.nan)
            results_df.insert(len(results_df.columns), f"p_{mod}_interaction", np.nan)

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        parcel = row["parcel"] 
        print("Parcel: ", parcel)
        
        # add intercept 
        data_df["intercept"] = 1

        # extract data needed to fit model & remove rows with incomplete data
        model_df = data_df[fixed_effects + [parcel] + [random_effects]]
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        n = len(np.unique(model_df["subject"]))
        row["n"] = n
        print(f"n = {n}")

        # fit model using statsmodels 
        model = sm.MixedLM(model_df[parcel], model_df[fixed_effects], 
                            groups = model_df[random_effects])
        
        model_fit = model.fit(method = ['cg'], reml=reml)
    
        print(model_fit.summary())

        # write out fitted model coefficients 
        row = write_model_fit(row, model_fit, fixed_effects, random_effects)

        # implement contrasts to test fixed effect of multi-level factors
        for factor in ml_factors: 
            if factor in moderators:
                continue 

            levels = code_map[factor]
            n_vars = len(levels)

            r_matrix = np.zeros(shape = (n_vars, len(model_fit.params)))

            for i, level in enumerate(levels):
                r_matrix[i, num_id[level]] = 1

            f_test = model_fit.f_test(r_matrix)
            row[f"F_{factor}"] = f_test.fvalue
            row[f"p_{factor}"] = f_test.pvalue

        # test overall effect of moderator, if included and a multi-level factor 
        if len(interaction_vars) > 1: 
            n_vars = len(interaction_vars)
            r_matrix = np.zeros(shape = (n_vars, len(model_fit.params)))

            for i, int in enumerate(interaction_vars): 
                r_matrix[i, num_id[int]] = 1

            f_test = model_fit.f_test(r_matrix)
            row[f"F_{mod}_interaction"] = f_test.fvalue
            row[f"p_{mod}_interaction"] = f_test.pvalue
        
        # test overall model significance
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        f_test = model_fit.f_test(r_matrix)
        row["F_model"] = f_test.fvalue
        row["p_model"] = f_test.pvalue

        results_df.iloc[idx] = row
    
    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def run_lme_model(parcel_df, clinical_df, predictor, outcomes, 
                  covariates=[], random_effects="subject", 
                  merge_on = ["subject", "session"], merge_ses=0, 
                  reml=False, out_df = None): 

    parcels = parcel_df.columns.to_list()
    parcels.remove("subject")
    parcels.remove("session")

    # only retain one predictor per parcel 
    # (e.g., only consider baseline parcel FC)
    if "session" not in merge_on:
        parcel_df = parcel_df[parcel_df.session == merge_ses]
        parcel_df = parcel_df.drop("session", axis=1)

    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = merge_on)
    
    if DEBUG:
        data_df.to_excel("debug/efield_fc_merged.xlsx")

    ### code factor levels as dummy variables ###

    # treatment paradigm
    data_df["tx_iTBS"] = data_df["tx_hz"].map(lambda x: 0 if x == 10 else 1)

    if "tx_type" in covariates: 
        covariates.remove("tx_type")
        covariates = covariates + ["tx_iTBS"]

    # session 
    data_df["sessionT5"] = data_df["session"].map(lambda x: 1 if x == 1 else 0)
    data_df["sessionEP"] = data_df["session"].map(lambda x: 1 if x == 2 else 0)

    # scan site
    data_df["siteMN"] = data_df["site_scan"].map(lambda x: 1 if x == 2000 else 0)
    data_df["sitePV"] = data_df["site_scan"].map(lambda x: 1 if x == 1976 else 0)
    data_df["siteWJ"] = data_df["site_scan"].map(lambda x: 1 if x == 1984 else 0)

    site_scan = False 

    if "site_scan" in covariates:
        site_scan = True
        covariates.remove("site_scan")
        covariates = ["siteMN", "sitePV", "siteWJ"] + covariates

    # add model intercept
    data_df["intercept"] = 1

    ### set up data frame to store results ### 
    results_df = pd.DataFrame(
        data = {"parcel" : np.tile(parcels, len(outcomes)), 
                "predictor" : predictor, 
                "covariates" : ", ".join(covariates),
                "outcome" : np.repeat(outcomes, len(parcels))}, 
        columns = ["parcel", "predictor", "outcome", "covariates", "model", 
                   "converged", "aic", "bic"])
        
    results_df.reset_index(drop=True, inplace=True)

    model_params = ["intercept", "parcel", "sessionT5", "sessionEP"] + covariates 
    
    for stat in ["coef", "se", "t", "p"]:
        for var in model_params: 
            results_df.insert(len(results_df.columns), f"{stat}_{var}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add test statistics & inferences for multi-level factors
    results_df["F_session"] = np.nan
    results_df["p_session"] = np.nan 

    if site_scan:
        # add test statistics & inferences for multi-level factors
        results_df["F_site"] = np.nan
        results_df["p_site"] = np.nan 

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():

        outcome = row["outcome"]
        parcel = row["parcel"]

        formula = f"{outcome} ~ 1 + {parcel} + sessionT5 + sessionEP"

        for cov in covariates: 
            formula = formula + f" + {cov}"

        for re in [random_effects]:
            formula = formula + f" + (1 | {re})"

        row["model"] = formula

        print("Running model: ", formula)

        model_df = data_df[[outcome, "intercept", parcel, "sessionT5", "sessionEP"] 
                           + covariates + [random_effects]]

        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        model = sm.MixedLM(model_df[outcome], model_df[model_params], 
                            groups = model_df[random_effects])
        
        model_fit = model.fit(method=['lbfgs'], reml=reml)
    
        for p in model_params:
            row[f"coef_{p}"] = model_fit.fe_params[p]
            row[f"se_{p}"] = model_fit.bse_fe[p]
            row[f"t_{p}"] = model_fit.tvalues[p]
            row[f"p_{p}"] = model_fit.pvalues[p]

        for i, re in enumerate([random_effects]):
            row[f"var_{re}"] = model_fit.cov_re.iloc[i,i]
        
        row["converged"] = model_fit.converged
        row["aic"] = model_fit.aic
        row["bic"] = model_fit.bic

        # implement contrasts to test fixed effect of session, scan site
        # test significance of session 
        r_matrix = np.zeros(shape = (2, len(model_fit.params)))
        r_matrix[0, 2] = 1
        r_matrix[1, 3] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_session"] = f_test.fvalue
        row["p_session"] = f_test.pvalue

        # test significance of scan site if included as a model parameter
        if site_scan:
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, 4] = 1
            r_matrix[1, 5] = 1
            r_matrix[2, 6] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_site"] = f_test.fvalue
            row["p_site"] = f_test.pvalue

        results_df.iloc[idx] = row
    
    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def generate_parser():
    parser = argparse.ArgumentParser(prog="model_efield_effects", 
            description="Generate linear mixed-effects models.")

    # required args 
    parser.add_argument("clinical_df", 
                        help = "File path to workbook containing all outcome variables & covariates.")
    parser.add_argument("out_fp", help = "Destination file path to store results.")

    # filter data 
    parser.add_argument("--filter_protocol", type=int, choices=[10, 50], 
                        help="Only retain subjects receiving specific TMS pulse frequency.")
    parser.add_argument("--filter_ptsd", type=int, choices=[0,1], 
                        help="Only retain subjects with PTSD (0) or without PTSD (1).")
    
    # optional args for model construction 
    parser.add_argument("--moderator_df", help = "Data frame storing a moderator of FC change")
    parser.add_argument("--atlases", nargs="+", 
                        help = "Workbook(s) containing values of fixed effects variables for atlas parcels.")
    parser.add_argument("--predictor", type=str, help="String description of fixed effects variable.")
    parser.add_argument("--outcomes", nargs="+", default=["qids_tot", "pcl5_tot", 
                                                          "vr36_emo_wellbeing", 
                                                          "vr36_energy", 
                                                          "vr36_role_limit_emo", 
                                                          "vr36_soc_fx"],
                         help = "Names of variables in clinical_df to model as outcomes.")
    parser.add_argument("--covariates", nargs="*", default=["site_scan", "tx_hz"],
                        help = "Names of variables in clinical_df to model as covariates.")
    parser.add_argument("--moderators", nargs="+", default=[], 
                        help = "Name of variables to model as moderator of network change." \
                        "Only relevant for modeling parcel_fc as outcome.")
    parser.add_argument("--tx_model", action='store_true', 
                        help = "Explicitly model changes at post-treatment time points.")
    parser.add_argument("--bl_predictor", action='store_true', 
                        help="Only uses baseline parcel data as a predictor "
                            "(as opposed to change over treatment)")
    parser.add_argument("--merge_on", nargs="+", default=["subject", "session"],
                        help="Which variables to merge predictor dataframe "
                        "and clinical dataframe on")
    parser.add_argument("--lm", action='store_true', 
                        help="Fit a standard linear regression using one timepoint only.")
    parser.add_argument("--session", type=int, choices=[1,2], 
                        help="Model effects for a specific session. " \
                            "Must be used with --lm option.")
    parser.add_argument("--covary_bl", action='store_true', 
                        help="Whether to include baseline outcome value as covariate")
    parser.add_argument("--perc_change", action='store_true',
                        help='Whether to model outcome change as a percentage rel. to baseline.')
    parser.add_argument("--norm", action='store_true', 
                        help="Normalize parcel predictor data")
    parser.add_argument("--norm_how", default="zscore", 
                        help = "Normalization method to use")

    # additional options
    parser.add_argument("--fdr_correct", action='store_true', 
                        help = "Correct parcel p-values for multiple comparisons by controlling FDR.")
    parser.add_argument("--verbose", action='store_true')

    return parser

if __name__ == "__main__": 

    parser = generate_parser()
    args = parser.parse_args()

    print("Atlases: ", args.atlases)

    clinical_df = pd.read_csv(args.clinical_df)
    clinical_df["subject"] = clinical_df["record_id"].map(redcap_to_bids)

    if args.filter_protocol is not None:
        hz = args.filter_protocol
        clinical_df = clinical_df.loc[clinical_df["tx_hz"] == hz]

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
    parcel_df = combine_atlases(args.atlases, args.merge_on)
    
    if "parcel_fc" in args.outcomes: 
        if args.moderator_df is not None:
            if "xlsx" in args.moderator_df:
                moderator_df = pd.read_excel(args.moderator_df)
            elif "csv" in args.moderator_df:
                moderator_df = pd.read_csv(args.moderator_df)
            else: 
                exit(1)

            if "record_id" in moderator_df.columns:
                moderator_df["subject"] = moderator_df["record_id"].map(redcap_to_bids)
            if not "TMS" in moderator_df["subject"].iat[0]:
                moderator_df["subject"] = moderator_df["subject"].map(redcap_to_bids)

        
        if args.tx_model:
            run_lme_fc_tx_model_2(parcel_df, clinical_df, 
                                covariates = args.covariates,
                                moderator_df = moderator_df,
                                moderators = args.moderators,
                                out_df = args.out_fp, 
                                covary_bl = args.covary_bl)
        else: 
            run_lme_fc_model_2(parcel_df, clinical_df, 
                                covariates = args.covariates,
                                moderator_df = moderator_df,
                                out_df = args.out_fp)
        
        args.outcomes.remove("parcel_fc")

    if len(args.outcomes) > 0:
        if args.tx_model:
            if args.lm:
                results_df = run_lm_tx_model(parcel_df, clinical_df, 
                                        args.outcomes, args.session,
                                        predictor=args.predictor,
                                        bl_predictor=args.bl_predictor,
                                        moderator=args.moderator,
                                        covariates=args.covariates,
                                        covary_bl = args.covary_bl,
                                        merge_on=args.merge_on, 
                                        out_df = args.out_fp, 
                                        fdr = args.fdr_correct, 
                                        verbose=args.verbose)
            else:
                results_df = run_lme_tx_model_2(parcel_df, clinical_df, 
                                            args.outcomes, 
                                            norm=args.norm, 
                                            norm_how=args.norm_how,
                                            predictor=args.predictor,
                                            covariates=args.covariates,
                                            bl_predictor=args.bl_predictor, 
                                            merge_on=args.merge_on, 
                                            out_df = args.out_fp, 
                                            fdr = args.fdr_correct, 
                                            verbose=args.verbose,
                                            covary_bl=args.covary_bl, 
                                            perc_change=args.perc_change)
        else:
            results_df = run_lme_model(parcel_df, clinical_df, 
                                    "parcel_fc", 
                                    args.outcomes, 
                                    covariates=args.covariates,
                                    out_df = args.out_fp)

    #results_df.to_excel(args.out_fp, index=False)