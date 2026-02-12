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

def combine_atlases(atlases):

    df = pd.read_excel(atlases[0])

    for atlas in atlases[1:]: 
        atlas = pd.read_excel(atlas)
        df = pd.merge(df, atlas, on = ["subject", "session"])
        
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

def compute_tx_change(data_df, change_vars): 
    ### compute change scores and remove rows for baseline session ##
    bl_df = data_df[data_df.session == 0]
    bl_df = bl_df.drop("session", axis=1)
    bl_df.columns = [col + "_bl" if col != "subject" else col 
                     for col in bl_df.columns]
    
    data_df = data_df[data_df.session != 0]

    data_df = pd.merge(data_df, bl_df, on = "subject")

    for var in change_vars: 
        data_df[f"{var}_change"] = (data_df[var] - data_df[f"{var}_bl"])

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
        factor_map["site_scan"] = [1991, "site"]

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
    parcels.remove("subject")
    parcels.remove("session")

    return parcels

def get_model_formula(predictor, covariates, random_effects): 
    # write out model formula  
    formula = f"parcel_fc ~ {predictor}"

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

    if "parcel_fc" in outcomes: 
        results_df = pd.DataFrame(
        data = {"parcel" : np.tile(parcels, len(moderators)), 
                "parcel_id" : np.tile(parcel_id, len(moderators)), 
                "outcome" : "parcel_fc", 
                "covariates" : ", ".join(covariates),
                "moderator" : np.repeat(moderators, len(parcels)), 
                "predictor" : predictor, 
                "model" : formula}, 
        columns = ["parcel", "parcel_id", "predictor", "outcome", "moderator", 
                   "covariates", "model", "n", "df", "reml", "converged", 
                   "aic", "bic"])
    else: 
        results_df = pd.DataFrame(
            data = {"parcel" : np.tile(parcels, len(outcomes)), 
                    "parcel_id" : np.tile(parcel_id, len(outcomes)), 
                    "outcome" : np.repeat(outcomes, len(parcels)), 
                    "covariates" : ", ".join(covariates),
                    "moderator" : "session", 
                    "predictor" : predictor, 
                    "model" : formula}, 
            columns = ["parcel", "parcel_id", "predictor", "outcome", "moderator", 
                       "covariates", "model","n", "df", "reml", "converged", 
                       "aic", "bic", "p_parcel", "F_parcel"])
    
    results_df.reset_index(drop=True, inplace=True)
    
    for stat in ["coef", "se", "t", "p"]:
        for fe in fixed_effects: 
            results_df.insert(len(results_df.columns), f"{stat}_{fe}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add test statistics & inferences for multi-level factors & overall model
    for p in ml_factors + moderators + ["model"]: 
        if f"F_{p}" not in results_df.columns:
            results_df.insert(len(results_df.columns), f"F_{p}", np.nan)
            results_df.insert(len(results_df.columns), f"p_{p}", np.nan)
    
    return results_df 

def write_model_fit(row, fit, fixed_effects, random_effects): 
    for fe in fixed_effects:
        row[f"coef_{fe}"] = fit.fe_params[fe]
        row[f"se_{fe}"] = fit.bse_fe[fe]
        row[f"t_{fe}"] = fit.tvalues[fe]
        row[f"p_{fe}"] = fit.pvalues[fe]

    for i, re in enumerate([random_effects]):
        row[f"var_{re}"] = fit.cov_re.iloc[i,i]
    
    row["converged"] = fit.converged
    row["aic"] = fit.aic
    row["bic"] = fit.bic

    return row 

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
                  predictor = "parcel_fc_change", 
                  covariates=[], random_effects="subject", 
                  ref_ses=0, 
                  merge_on = ["subject", "session"], 
                  reml=False, fdr=False, out_df = None, 
                  verbose=False): 

    parcels = get_parcels(parcel_df)

    # only retain variables needed for model
    clinical_df = clinical_df[["subject", "session"] + outcomes + covariates]

    # merge predictor and clinical outcome data 
    data_df = pd.merge(clinical_df, 
                       parcel_df, 
                       on = merge_on)

    data_df.to_excel("debug/lme_tx_model_data_orig.xlsx", index=False)

    ### compute change scores and remove rows for baseline session ##

    data_df = compute_tx_change(data_df, parcels + outcomes)
    data_df.to_excel("debug/lme_tx_model_data_change.xlsx", index=False)

    ### code factor levels as dummy variables ###
    predictors = ["session"] + covariates
    factor_map = create_factor_map(predictors, ref_ses)
    data_df, ml_factors, code_map = code_factors(data_df, factor_map)

    # add parcel_fc_change as a predictor if it is not being coded as two distinct interactions 
    if ref_ses in [1,2]: 
        predictors += ["parcel"]

    # concatenate list of dummy regressors coded for each factor variable
    factor_vars = list(code_map.keys())
    dummy_vars = []
    for vars in code_map.values(): 
        dummy_vars = dummy_vars + vars

    # identify continuous predictors  
    continuous_vars = predictors.copy()
    for x in factor_vars: 
        continuous_vars.remove(x)

    # summarize types of predictor included in model 
    print("Factor variables: ", factor_vars)
    print("Continuous predictors: ", continuous_vars)

    ### specify session x parcel interaction terms ### 
    interaction_vars = []
    ses_vars = code_map["session"] 

    for ses in ses_vars:
        interaction_vars.append("interaction_" + ses + "_parcel")

    ### finally, specify all explicitly coded fixed effect regressors ### 
    fixed_effects = continuous_vars + dummy_vars + interaction_vars 
    print("Fixed effects: ", fixed_effects)

    num_id = dict(zip(fixed_effects, range(len(fixed_effects)))) 

    ### set up data frame to store results ### 

    results_df = setup_results_df(parcels, predictor, outcomes, 
                                  covariates, fixed_effects, random_effects, 
                                  ml_factors)
    
    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        outcome = row["outcome"]
        parcel = row["parcel"]

        print("Parcel: ", parcel)
        
        # standardize parcel predictor name & outcome name for convenience 
        data_df["outcome"] = data_df[f"{outcome}_change"]
        data_df["parcel"] = data_df[f"{parcel}_change"]
        
        # compute interaction terms
        for int in interaction_vars: 
            ses = int.split("_")[1]
            data_df[int] = data_df[ses] * data_df["parcel"]

        # extract model regressors  
        model_df = data_df[fixed_effects + ["outcome"] + [random_effects]]

        # remove rows with missing model variables 
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)
        row["df"] = len(model_df) 
        row["n"] = len(np.unique(model_df["subject"]))

        # construct model 
        model = sm.MixedLM(model_df["outcome"], model_df[fixed_effects], 
                            groups = model_df[random_effects])
        
        # fit model
        try:
            model_fit = model.fit(method=['cg'], reml=reml)
        except Exception as e:
            print(f"Fitting failed with error message: {e}")
            
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

        f_test = model_fit.f_test(r_matrix)
        row["F_model"] = f_test.fvalue
        row["p_model"] = f_test.pvalue

        # implement contrasts to test main effect of parcel fc change 
        r_matrix = np.zeros(shape = (len(interaction_vars), len(model_fit.params)))

        for i, int in enumerate(interaction_vars):
            r_matrix[i, num_id[int]] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_parcel"] = f_test.fvalue
        row["p_parcel"] = f_test.pvalue

        # implement contrasts to test fixed effect of session
        row = test_multi_level_factor_effects(ml_factors, code_map, model_fit, 
                                              num_id, row)   

        results_df.iloc[idx] = row

    # FDR correction using Benjamini/Hochberg method
    if fdr: 
        _, p_corr, _, _ = multitest.multipletests(results_df["p_parcel"], method = 'fdr_bh')
        results_df["p_parcel_corr"] = p_corr

    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def run_lme_tx_model(parcel_df, clinical_df, outcomes, 
                  predictor = "parcel_fc_change", 
                  covariates=[], random_effects="subject", 
                  bl_severity=False, 
                  merge_on = ["subject", "session"], 
                  reml=False, fdr=False, out_df = None, 
                  verbose=False): 

    parcels = parcel_df.columns.to_list()
    parcels.remove("subject")
    parcels.remove("session")
    parcel_id = np.arange(len(parcels))

    # only retain variables needed for model
    clinical_df = clinical_df[["subject", "session"] + outcomes + covariates]

    # merge predictor and clinical outcome data 
    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = merge_on)

    ### compute change scores and remove rows for baseline session ##
    bl_df = data_df.loc[data_df.session == 0, ["subject"] + parcels + outcomes]
    #bl_df = bl_df.drop("session", axis=1)
    bl_df.columns = [col + "_bl" if col != "subject" else col for col in bl_df.columns]
    
    data_df = data_df[data_df.session != 0]

    if DEBUG:
        bl_df.to_excel("debug/efield_clinical_bl_merged.xlsx")

    data_df = pd.merge(data_df, bl_df, on = "subject")
    change_df = data_df[["subject", "session"] + covariates].copy()

    for outcome in outcomes: 
        data_df[f"{outcome}_change"] = (data_df[outcome] - data_df[f"{outcome}_bl"])
        change_df[f"{outcome}_change"] = (data_df[outcome] - data_df[f"{outcome}_bl"])

    for parcel in parcels: 
        data_df[f"{parcel}_change"] = (data_df[parcel] - data_df[f"{parcel}_bl"])
        change_df[f"{parcel}_change"] = (data_df[parcel] - data_df[f"{parcel}_bl"])

    if DEBUG:
        change_df.to_excel("debug/lme_tx_model_data.xlsx")

    ### code factor levels as dummy variables ###

    # session 
    data_df["sessionT5"] = data_df["session"].map(lambda x: 1 if x == 1 else 0)
    
    # treatment paradigm
    if "tx_hz" in covariates: 
        data_df["tx_iTBS"] = data_df["tx_hz"].map(lambda x: 0 if x == 10 else 1)
 
        covariates.remove("tx_hz")
        covariates = covariates + ["tx_iTBS"]

    # scan site
    site_scan = None 

    if "site_scan" in covariates:
        data_df["siteMN"] = data_df["site_scan"].map(lambda x: 1 if x == 2000 else 0)
        data_df["sitePV"] = data_df["site_scan"].map(lambda x: 1 if x == 1976 else 0)
        data_df["siteWJ"] = data_df["site_scan"].map(lambda x: 1 if x == 1984 else 0)

        covariates.remove("site_scan")
        site_scan = len(covariates)
        covariates = covariates + ["siteMN", "sitePV", "siteWJ"] 

    ### set up data frame to store results ### 

    results_df = pd.DataFrame(
        data = {"parcel" : np.tile(parcels, len(outcomes)), 
                "parcel_id" : np.tile(parcel_id, len(outcomes)), 
                "predictor" : predictor, 
                "covariates" : ", ".join(covariates),
                "outcome" : np.repeat(outcomes, len(parcels)), 
                "reml" : reml}, 
        columns = ["parcel", "parcel_id", "predictor", "outcome", "covariates", "model",
                   "n", "df", "reml", "converged", "aic", "bic"])
        
    results_df.reset_index(drop=True, inplace=True)

    model_params = ["intercept", "sessionT5", "parcelEP", 
                    "interactionT5"] + covariates 
    
    if bl_severity: 
        model_params += ["severityBL"]

    num_id = dict(zip(model_params, range(len(model_params))))
    
    for stat in ["coef", "se", "t", "p"]:
        for var in model_params: 
            results_df.insert(len(results_df.columns), f"{stat}_{var}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add F statistic for overall model significance 
    results_df["F_model"] = np.nan
    results_df["p_model"] = np.nan

    # add F statistic for main effect of parcel FC change 
    results_df["F_parcel"] = np.nan
    results_df["p_parcel"] = np.nan

    # add F statistic for main effect of session 
    results_df["F_session"] = np.nan
    results_df["p_session"] = np.nan 

    # add test statistics & inferences for multi-level factors
    if site_scan is not None:
        results_df["F_site"] = np.nan
        results_df["p_site"] = np.nan 

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():

        outcome = row["outcome"]
        parcel = row["parcel"]

        formula = f"{outcome}_change ~ 1 + session x {parcel}_change"

        if bl_severity: 
            formula += f" + {outcome}_bl"
        
        for cov in covariates: 
            formula = formula + f" + {cov}"

        for re in [random_effects]:
            formula = formula + f" + (1 | {re})"

        row["model"] = formula

        if verbose:
            print("Fitting model: ", formula)

        model_df = data_df[[f"{parcel}_change", f"{outcome}_change", "sessionT5"]  + 
                           covariates + [random_effects]]

        # add model intercept
        model_df["intercept"] = 1

        # if baseline symptom severity is a covariate, add now 
        if bl_severity:
            model_df["severityBL"] = data_df[f"{outcome}_bl"]
        
        # standardize predictor and outcome variable names for convenience 
        model_df["outcome"] = model_df[f"{outcome}_change"]
        model_df["parcelEP"] = model_df[f"{parcel}_change"]

        # compute interaction term 
        model_df["interactionT5"] = model_df["sessionT5"] * model_df["parcelEP"]

        # remove rows with missing model variables 
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)
        row["df"] = len(model_df) 
        row["n"] = len(np.unique(model_df["subject"]))

        # construct model 
        model = sm.MixedLM(model_df["outcome"], model_df[model_params], 
                            groups = model_df["subject"])
        
        # fit model

        try:
            model_fit = model.fit(method=['cg'], reml=reml)
        except Exception as e:
            print(f"Model fit failed with error message: {e}.\n")
            
            row["converged"] = False 
            results_df.iloc[idx] = row

            continue
    
        if verbose:
            print(model_fit.summary())

        # write out parameters of fitted model 
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

        # implement contrasts to test overall model fit
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        f_test = model_fit.f_test(r_matrix)
        row["F_model"] = f_test.fvalue
        row["p_model"] = f_test.pvalue

        # implement contrasts to test main effect of parcel fc change 
        r_matrix = np.zeros(shape = (2, len(model_fit.params)))
        r_matrix[0, num_id["parcelEP"]] = 1
        r_matrix[1, num_id["interactionT5"]] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_parcel"] = f_test.fvalue
        row["p_parcel"] = f_test.pvalue

        # implement contrasts to test fixed effect of session
        r_matrix = np.zeros(shape = (2, len(model_fit.params)))
        r_matrix[0, num_id["sessionT5"]] = 1
        r_matrix[1, num_id["interactionT5"]] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_session"] = f_test.fvalue
        row["p_session"] = f_test.pvalue

        # test significance of scan site if included as a model parameter
        if site_scan:
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, num_id["siteMN"]] = 1
            r_matrix[1, num_id["sitePV"]] = 1
            r_matrix[2, num_id["siteWJ"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_site"] = f_test.fvalue
            row["p_site"] = f_test.pvalue

        results_df.iloc[idx] = row

    # FDR correction using Benjamini/Hochberg method
    if fdr: 
        _, p_corr, _, _ = multitest.multipletests(results_df["p_parcel"], method = 'fdr_bh')
        results_df["p_parcel_corr"] = p_corr

    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def run_bl_lme_tx_model(parcel_df, clinical_df, outcomes, 
                  predictor = "parcel_fc_bl", 
                  covariates=[], random_effects="subject", 
                  reml=False, fdr=False, out_df = None, 
                  verbose=False): 

    parcels = parcel_df.columns.to_list()
    parcels.remove("subject")
    parcels.remove("session")
    parcel_id = np.arange(1, len(parcels)+1) # 1-indexing to match Schaefer atlas

    # only retain baseline FC data 
    parcel_df = parcel_df[parcel_df.session == 0]
    parcel_df.drop(columns = "session", inplace=True)

    # only retain clinical variables needed for model
    clinical_df = clinical_df[["subject", "session"] + outcomes + covariates]

    # merge predictor and clinical outcome data 
    data_df = pd.merge(clinical_df, 
                       parcel_df, 
                       on = "subject")

    ### compute change scores and remove rows for baseline session ##
    bl_df = data_df.loc[data_df.session == 0, ["subject"] + outcomes]
    bl_df.columns = [col + "_bl" if col != "subject" else col for col in bl_df.columns]
    
    data_df = data_df[data_df.session != 0]

    if DEBUG:
        bl_df.to_excel("debug/efield_clinical_bl_merged.xlsx")

    data_df = pd.merge(data_df, bl_df, on = "subject")

    for outcome in outcomes: 
        data_df[f"{outcome}_change"] = (data_df[outcome] - data_df[f"{outcome}_bl"])

    if DEBUG:
        data_df.to_excel("debug/efield_bl_lme_tx_model_data.xlsx")

    ### code factor levels as dummy variables ###

    # session 
    data_df["sessionT5"] = data_df["session"].map(lambda x: 1 if x == 1 else 0)
    
    # treatment paradigm
    if "tx_hz" in covariates: 
        data_df["tx_iTBS"] = data_df["tx_hz"].map(lambda x: 0 if x == 10 else 1)
 
        covariates.remove("tx_hz")
        covariates = covariates + ["tx_iTBS"]

    # scan site
    site_scan = None 

    if "site_scan" in covariates:
        data_df["siteMN"] = data_df["site_scan"].map(lambda x: 1 if x == 2000 else 0)
        data_df["sitePV"] = data_df["site_scan"].map(lambda x: 1 if x == 1976 else 0)
        data_df["siteWJ"] = data_df["site_scan"].map(lambda x: 1 if x == 1984 else 0)

        covariates.remove("site_scan")
        site_scan = len(covariates)
        covariates = covariates + ["siteMN", "sitePV", "siteWJ"] 

    ### set up data frame to store results ### 

    results_df = pd.DataFrame(
        data = {"parcel" : np.tile(parcels, len(outcomes)), 
                "parcel_id" : np.tile(parcel_id, len(outcomes)), 
                "predictor" : predictor, 
                "covariates" : ", ".join(covariates),
                "outcome" : np.repeat(outcomes, len(parcels)), 
                "reml" : reml}, 
        columns = ["parcel", "parcel_id", "predictor", "outcome", "covariates", "model",
                   "n_obs", "n_groups", "reml", "converged", "aic", "bic"])
        
    results_df.reset_index(drop=True, inplace=True)

    model_params = ["intercept", "sessionT5", "parcelEP", 
                    "interactionT5"] + covariates 
    
    num_id = dict(zip(model_params, range(len(model_params))))
    
    for stat in ["coef", "se", "t", "p"]:
        for var in model_params: 
            results_df.insert(len(results_df.columns), f"{stat}_{var}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add F statistic for overall model significance 
    results_df["F_model"] = np.nan
    results_df["p_model"] = np.nan

    # add F statistic for main effect of parcel FC change 
    results_df["F_parcel"] = np.nan
    results_df["p_parcel"] = np.nan

    # add F statistic for main effect of session 
    results_df["F_session"] = np.nan
    results_df["p_session"] = np.nan 

    # add test statistics & inferences for multi-level factors

    if site_scan is not None:
        results_df["F_site"] = np.nan
        results_df["p_site"] = np.nan 

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():

        outcome = row["outcome"]
        parcel = row["parcel"]

        formula = f"{outcome}_change ~ 1 + session x {predictor}"

        for cov in covariates: 
            formula = formula + f" + {cov}"

        for re in [random_effects]:
            formula = formula + f" + (1 | {re})"

        row["model"] = formula

        if verbose:
            print("Fitting model: ", formula)

        model_df = data_df[[parcel, f"{outcome}_change", "sessionT5"]  + 
                           covariates + [random_effects]]

        # add model intercept
        model_df["intercept"] = 1

        # standardize predictor and outcome variable names for convenience 
        model_df["outcome"] = model_df[f"{outcome}_change"]
        model_df["parcelEP"] = model_df[parcel]

        # compute interaction term 
        model_df["interactionT5"] = model_df["sessionT5"] * model_df["parcelEP"]

        # remove rows with missing model variables 
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        # construct model 
        model = sm.MixedLM(model_df["outcome"], model_df[model_params], 
                            groups = model_df["subject"])
        
        # fit model

        try:
            model_fit = model.fit(method=['cg'], reml=reml)
        except Exception as e:
            print(f"Model fit failed with error message: {e}")
            
            row["converged"] = False 
            results_df.iloc[idx] = row

            continue
    
        if verbose:
            print(model_fit.summary())

        # write out parameters of fitted model 
        for p in model_params:
            row[f"coef_{p}"] = model_fit.fe_params[p]
            row[f"se_{p}"] = model_fit.bse_fe[p]
            row[f"t_{p}"] = model_fit.tvalues[p]
            row[f"p_{p}"] = model_fit.pvalues[p]

        for i, re in enumerate([random_effects]):
            row[f"var_{re}"] = model_fit.cov_re.iloc[i,i]
        
        # write out overal model fit properties
        row["n_obs"] = len(model_df) 
        row["n_groups"] = len(np.unique(model_df["subject"]))
        row["converged"] = model_fit.converged
        row["aic"] = model_fit.aic
        row["bic"] = model_fit.bic

        # implement contrasts to test overall model fit
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        f_test = model_fit.f_test(r_matrix)
        row["F_model"] = f_test.fvalue
        row["p_model"] = f_test.pvalue

        # implement contrasts to test main effect of parcel fc change 
        r_matrix = np.zeros(shape = (2, len(model_fit.params)))
        r_matrix[0, num_id["parcelEP"]] = 1
        r_matrix[1, num_id["interactionT5"]] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_parcel"] = f_test.fvalue
        row["p_parcel"] = f_test.pvalue

        # implement contrasts to test fixed effect of session
        r_matrix = np.zeros(shape = (2, len(model_fit.params)))
        r_matrix[0, num_id["sessionT5"]] = 1
        r_matrix[1, num_id["interactionT5"]] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_session"] = f_test.fvalue
        row["p_session"] = f_test.pvalue

        # test significance of scan site if included as a model parameter
        if site_scan:
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, num_id["siteMN"]] = 1
            r_matrix[1, num_id["sitePV"]] = 1
            r_matrix[2, num_id["siteWJ"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_site"] = f_test.fvalue
            row["p_site"] = f_test.pvalue

        results_df.iloc[idx] = row

    # FDR correction using Benjamini/Hochberg method
    if fdr: 
        _, p_corr, _, _ = multitest.multipletests(results_df["p_parcel"], method = 'fdr_bh')
        results_df["p_parcel_corr"] = p_corr

    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df

def run_lm_tx_model(parcel_df, clinical_df, outcomes, session, 
                  predictor = "parcel_fc", 
                  covariates=[],  
                  merge_on = ["subject", "session"], 
                  fdr=False, out_df = None, verbose=False): 

    parcels = parcel_df.columns.to_list()
    parcels.remove("subject")
    parcels.remove("session")
    parcel_id = np.arange(len(parcels))

    # only retain variables needed for model
    clinical_df = clinical_df[["subject", "session"] + outcomes + covariates]

    # merge predictor and clinical outcome data 
    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = merge_on)

    ### compute change scores and remove rows for baseline session ##
    bl_df = data_df.loc[data_df.session == 0, ["subject"] + parcels + outcomes]
    bl_df.columns = [col + "_bl" if col != "subject" else col for col in bl_df.columns]
    
    data_df = data_df[data_df.session == session]

    if DEBUG:
        bl_df.to_excel("debug/efield_clinical_bl_merged.xlsx")

    data_df = pd.merge(data_df, bl_df, on = "subject")
    change_df = data_df[["subject", "session"] + covariates].copy()

    for outcome in outcomes: 
        data_df[f"{outcome}_change"] = data_df[outcome] - data_df[f"{outcome}_bl"]
        change_df[f"{outcome}_change"] = data_df[outcome] - data_df[f"{outcome}_bl"]

    for parcel in parcels: 
        data_df[f"{parcel}_change"] = data_df[parcel] - data_df[f"{parcel}_bl"]
        change_df[f"{parcel}_change"] = data_df[parcel] - data_df[f"{parcel}_bl"]

    if DEBUG:
        change_df.to_excel("debug/efield_clinical_change_merged.xlsx")

    ### code factor levels as dummy variables ###

    # treatment paradigm
    if "tx_hz" in covariates: 
        data_df["tx_iTBS"] = data_df["tx_hz"].map(lambda x: 0 if x == 10 else 1)
 
        covariates.remove("tx_hz")
        covariates = covariates + ["tx_iTBS"]

    # scan site
    site_scan = False 

    if "site_scan" in covariates:
        site_scan = True

        data_df["siteMN"] = data_df["site_scan"].map(lambda x: 1 if x == 2000 else 0)
        data_df["sitePV"] = data_df["site_scan"].map(lambda x: 1 if x == 1976 else 0)
        data_df["siteWJ"] = data_df["site_scan"].map(lambda x: 1 if x == 1984 else 0)

        covariates.remove("site_scan")
        site_scan = len(covariates)
        covariates = covariates + ["siteMN", "sitePV", "siteWJ"] 

    ### set up data frame to store results ### 

    results_df = pd.DataFrame(
        data = {"parcel" : np.tile(parcels, len(outcomes)), 
                "parcel_id" : np.tile(parcel_id, len(outcomes)), 
                "predictor" : predictor, 
                "session" : session, 
                "covariates" : ", ".join(covariates),
                "outcome" : np.repeat(outcomes, len(parcels))}, 
        columns = ["parcel", "parcel_id", "predictor", "outcome", "covariates", "model",
                   "n", "df", "rsquared", "adj_rsquared", "F_model", "p_model"])
        
    results_df.reset_index(drop=True, inplace=True)

    model_params = ["intercept", "parcel"] + covariates 
    
    num_id = dict(zip(model_params, range(len(model_params))))
    
    for stat in ["coef", "se", "t", "p"]:
        for var in model_params: 
            results_df.insert(len(results_df.columns), f"{stat}_{var}", np.nan)

    # add test statistics & inferences for multi-level factors
    if site_scan:
        results_df["F_site"] = np.nan
        results_df["p_site"] = np.nan 

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():

        outcome = row["outcome"]
        parcel = row["parcel"]

        formula = f"{outcome}_change ~ 1 + {parcel}_change"

        for cov in covariates: 
            formula = formula + f" + {cov}"

        row["model"] = formula

        if verbose:
            print("Fitting model: ", formula)

        model_df = data_df[[f"{parcel}_change", f"{outcome}_change"]  + 
                           covariates]

        # add model intercept
        model_df["intercept"] = 1

        # standardize predictor and outcome variable names for convenience 
        model_df["outcome"] = model_df[f"{outcome}_change"]
        model_df["parcel"] = model_df[f"{parcel}_change"]

        # remove rows with missing model variables 
        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        # construct model 
        model = sm.OLS(model_df["outcome"], model_df[model_params])
        
        # fit model
        try:
            model_fit = model.fit()
        except Exception as e:
            print(f"Model fit failed with error message: {e}")
            
            row["converged"] = False 
            results_df.iloc[idx] = row

            continue
    
        if verbose:
            print(model_fit.summary())

        # write out parameters of fitted model 
        for p in model_params:
            row[f"coef_{p}"] = model_fit.params[p]
            row[f"se_{p}"] = model_fit.bse[p]
            row[f"t_{p}"] = model_fit.tvalues[p]
            row[f"p_{p}"] = model_fit.pvalues[p]
        
        # write out overall model properties
        row["rsquared"] = model_fit.rsquared
        row["adj_rsquared"] = model_fit.rsquared_adj
        row["F_model"] = model_fit.fvalue
        row["p_model"] = model_fit.f_pvalue
        row["df"] = model_fit.df_model
        row["n"] = model_fit.nobs

        # test significance of scan site using one-way ANOVA 
        # if included as a model parameter
        if site_scan:
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, num_id["siteMN"]] = 1
            r_matrix[1, num_id["sitePV"]] = 1
            r_matrix[2, num_id["siteWJ"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_site"] = f_test.fvalue
            row["p_site"] = f_test.pvalue

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

def run_lme_fc_tx_model(parcel_df, clinical_df, cluster_df=None, covariates=[], 
                    biotype=False, random_effects="subject", reml=False, 
                    out_df = None):
    
    parcels = parcel_df.columns.to_list()
    parcels.remove("subject")
    parcels.remove("session")

    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = ["subject", "session"])
    
    cluster=False
    clusters=[]
    if cluster_df is not None: 
        data_df = pd.merge(data_df, 
                           cluster_df[["subject", "cluster_label"]], 
                           on = ["subject"])
        cluster=True 

    if DEBUG:
        data_df.to_excel("debug/parcel_fc_merged.xlsx")

    ### compute change scores and remove rows for baseline session ##
    bl_df = data_df[data_df.session == 0]
    bl_df = bl_df.drop("session", axis=1)
    bl_df.columns = [col + "_bl" if col != "subject" else col 
                     for col in bl_df.columns]
    
    data_df = data_df[data_df.session != 0]

    if DEBUG:
        bl_df.to_excel("debug/efield_fc_bl_merged.xlsx")

    data_df = pd.merge(data_df, bl_df, on = "subject")

    for parcel in parcels: 
        data_df[f"{parcel}_change"] = (data_df[parcel] - data_df[f"{parcel}_bl"])

    ### code factor levels as dummy variables ###

    # store multi-level factors for ANOVA comparisons
    ml_factors = ["session"]

    # treatment paradigm
    data_df["tx_iTBS"] = data_df["tx_hz"].map(lambda x: 1 if x == 50 else 0)

    if "tx_hz" in covariates: 
        covariates.remove("tx_hz")
        covariates = covariates + ["tx_iTBS"]

    # session 
    data_df["sessionEP"] = data_df["session"].map(lambda x: 1 if x == 2 else 0)

    # scan site
    if "site_scan" in covariates:
        data_df["siteMN"] = data_df["site_scan"].map(lambda x: 1 if x == 2000 else 0)
        data_df["sitePV"] = data_df["site_scan"].map(lambda x: 1 if x == 1976 else 0)
        data_df["siteWJ"] = data_df["site_scan"].map(lambda x: 1 if x == 1984 else 0)

        covariates.remove("site_scan")
        covariates = covariates + ["siteMN", "sitePV", "siteWJ"] 

        ml_factors = ml_factors + ["site_scan"]

    if biotype:
        data_df["biotypeC+"] = data_df["biotype"].map(lambda x: 1 if x == "C+ biotype" else 0)
        ml_factors = ml_factors + ["biotype"]

    # clusters describing pre-treatment stim site network connectivity
    if cluster:
        num_clusters = len(np.unique(data_df["cluster_label"]))

        for i in range(1, num_clusters):
            data_df[f"cluster{i}"] = data_df["cluster_label"].map(lambda x: 1 if x == i else 0)
            clusters.append(f"cluster{i}")
        
        if num_clusters > 2:
            ml_factors = ml_factors + ["cluster", "cluster_int"]

    ### specify model regressors ###
    model_params = ["intercept", "sessionEP"] 

    if biotype:
        model_params += ["biotypeC+", "interactionEPxC+"]

    if cluster: 
        model_params = model_params + clusters

        for ses in ["T5", "EP"]: 
            for i in range(1, num_clusters):
                model_params.append(f"interaction{i}{ses}")

    model_params = model_params + covariates 
    
    num_id = dict(zip(model_params, range(len(model_params)))) 

    if cluster:
        predictor = "cluster x session"
    elif biotype:
        predictor = "biotype x session"
    else:
        predictor = "session"

    ### set up data frame to store results ### 
    results_df = pd.DataFrame(
        data = {"parcel" : parcels, 
                "predictor" : predictor,
                "outcome" : "parcel_fc",
                "covariates" : ", ".join(covariates)}, 
        columns = ["parcel", "predictor", "outcome", "covariates", "model", 
                   "converged", "aic", "bic"])
        
    results_df.reset_index(drop=True, inplace=True)
    
    for stat in ["coef", "se", "t", "p"]:
        for var in model_params: 
            results_df.insert(len(results_df.columns), f"{stat}_{var}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add test statistics & inferences for multi-level factors & overall model
    for p in ml_factors + ["model"]: 
        results_df.insert(len(results_df.columns), f"F_{p}", np.nan)
        results_df.insert(len(results_df.columns), f"p_{p}", np.nan)

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        parcel = row["parcel"] + "_change"

        if cluster: 
            formula = "parcel_fc ~ 1 + cluster x session"
        elif biotype: 
            formula = "parcel_fc ~ 1 + biotype x session"
        else:
            formula = f"parcel_fc ~ 1 + session"

        for cov in covariates: 
            formula = formula + f" + {cov}"

        for re in [random_effects]:
            formula = formula + f" + (1 | {re})"

        row["model"] = formula

        print("Running model: ", formula)

        # primary terms, i.e. not including interactions and intercept to be added
        terms = [parcel, "sessionEP"] + clusters + covariates + [random_effects]

        if biotype: 
            terms += ["biotypeC+"]

        model_df = data_df[terms]

        # add model intercept
        model_df["intercept"] = 1

        # add interactions 
        if cluster: 
            for i in range(1, num_clusters):
                model_df[f"interaction{i}EP"] = model_df[f"sessionEP"]*model_df[f"cluster{i}"]

        elif biotype: 
            model_df["interactionEPxC+"] = model_df["biotypeC+"]*model_df["sessionEP"]

        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        n = len(np.unique(model_df["subject"]))
        print(f"\tn = {n}")

        if DEBUG:
            model_df.to_excel("debug/parcel_change_model_data.xlsx",
                              index=False)

        model = sm.MixedLM(model_df[parcel], model_df[model_params], 
                            groups = model_df[random_effects])
        
        model_fit = model.fit(method = ['cg'], reml=reml)
    
        print(model_fit.summary())
        
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

        # implement contrasts to test fixed effect of multi-level factors
        n_vars = 2 + 1*biotype
        r_matrix = np.zeros(shape = (n_vars, len(model_fit.params)))
        
        # test joint hypothesis that both intercept (FC change at T5) = 0, and 
        # sessionEP factor coefficient (FC change at EP) = 0
        r_matrix[0, num_id["intercept"]] = 1
        r_matrix[1, num_id["sessionEP"]] = 1

        if biotype: 
            r_matrix[2, num_id["interactionEPxC+"]] = 1
        
        f_test = model_fit.f_test(r_matrix)
        row["F_session"] = f_test.fvalue
        row["p_session"] = f_test.pvalue

        # test significance of biotype if included as a model parameter 
        if biotype: 
            r_matrix = np.zeros(shape = (2, len(model_fit.params)))
            r_matrix[0, num_id["biotypeC+"]] = 1
            r_matrix[1, num_id["interactionEPxC+"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_biotype"] = f_test.fvalue
            row["p_biotype"] = f_test.pvalue

        # test significance of scan site if included as a model parameter
        if "site_scan" in ml_factors:
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, num_id["siteMN"]] = 1
            r_matrix[1, num_id["sitePV"]] = 1
            r_matrix[2, num_id["siteWJ"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_site_scan"] = f_test.fvalue
            row["p_site_scan"] = f_test.pvalue
        
        # test significance of clusters if included as a model parameter
        if "cluster" in ml_factors:
            r_matrix = np.zeros(shape = (num_clusters-1, len(model_fit.params)))

            for i in range(1, num_clusters):
                r_matrix[i-1, num_id[f"cluster{i}"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_cluster"] = f_test.fvalue
            row["p_cluster"] = f_test.pvalue
        
        # test significance of cluster interactions if included as a model parameter
        if "cluster" in ml_factors:
            r_matrix = np.zeros(shape = (num_clusters-1, len(model_fit.params)))

            for i in range(1, num_clusters):
                r_matrix[i-1, num_id[f"interaction{i}EP"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_cluster_int"] = f_test.fvalue
            row["p_cluster_int"] = f_test.pvalue

        # test significance of overall model 
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        f_test = model_fit.f_test(r_matrix)
        row["F_model"] = f_test.fvalue
        row["p_model"] = f_test.pvalue

        results_df.iloc[idx] = row
    
    if out_df is not None:
        results_df.to_excel(out_df, index=False)

    return results_df


def run_lme_fc_tx_model_2(parcel_df, clinical_df, moderator_df=None, 
                          covariates=[], moderator=None,
                         random_effects="subject", reml=False,
                        include_bl = True, out_df = None):
    
    # get list of parcels in data frame 
    parcels = get_parcels(parcel_df)

    # only retain variables included in model from clinical dataset
    clinical_df = clinical_df[["subject", "session"] + covariates]

    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = ["subject", "session"])
    
    # add moderators 
    if not (moderator_df is None or moderator is None): 
        data_df = pd.merge(data_df, 
                           moderator_df[["subject", moderator]], 
                           on = ["subject"], 
                           how = "outer")

    # compile variables to be included as predictors in models 
    predictors = ["session"] + covariates

    ### compute delta values and remove rows for baseline session ##
    data_df = compute_tx_change(data_df, parcels)

    if DEBUG: 
        data_df.to_excel(f"debug/fc_change_data_mod_{moderator}.xlsx", index=False)

    ### code factor levels as dummy variables ###
    # identify which of the modeled variables are factor variables
    factor_map = create_factor_map(predictors + [moderator])
    factor_vars = list(factor_map.keys())
    print("Factor variables: ", factor_vars)
   
    # code factor variables using dummy variables 
    data_df, ml_factors, code_map = code_factors(data_df, factor_map)

    # concatenate list of dummy regressors coded for each factor variable
    # EXCLUDE moderator terms, which will be included as interactions
    dummy_vars = []
    for factor, vars in code_map.items(): 
        if factor != moderator: 
            dummy_vars = dummy_vars + vars

    # identify continuous variables 
    continuous_vars = predictors.copy()
    for x in factor_vars: 
        if x != moderator: 
            continuous_vars.remove(x)

    print("Continuous predictors: ", continuous_vars)

    ### compute interaction terms ### 
    interaction_vars = []
    ses_vars = code_map["session"] 
    print("Session variables: ", ses_vars)
    print("Moderator: ", moderator)

    for ses in ses_vars: 
        if moderator is None: 
            print("Model does not contain a moderator!")
            break

        if moderator in factor_vars: 
            print("Moderator is a factor variable!")
            for level in code_map[moderator]:
                var_name = "interaction_" + ses + "_" + level

                data_df[var_name] = data_df[ses] * data_df[level]

                interaction_vars.append(var_name)
            
        else: 
            print("Moderator is a continuous variable!")

            var_name = "interaction_" + ses + "_" + moderator

            data_df[var_name] = data_df[ses] * data_df[moderator]

            interaction_vars.append(var_name)

    print("Interaction variables: ", interaction_vars)

    ### specify explicitly coded fixed effect regressors ###
    if include_bl: 
        continuous_vars += ["parcel_bl"]
    
    fixed_effects = continuous_vars + dummy_vars + interaction_vars
    num_id = dict(zip(fixed_effects, range(len(fixed_effects)))) 

    print("Fixed effect regressors: ", fixed_effects)
        
    # write out fixed effect structure   
    if moderator is not None:        
        predictor = moderator + " x session"
    else:
        predictor = "session"

    ### set up data frame to store results ### 
    results_df = setup_results_df(parcels, predictor, ["parcel_fc"], covariates,
                                  fixed_effects, random_effects, 
                                  ml_factors, [moderator])

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

        if moderator in test_factors: 
            test_factors.remove(moderator)

        row = test_multi_level_factor_effects(test_factors, code_map, model_fit, 
                                              num_id, row)

        # test overall effect of moderator, if included and a multi-level factor 
        if len(interaction_vars) > 1: 
            n_vars = len(interaction_vars)
            r_matrix = np.zeros(shape = (n_vars, len(model_fit.params)))

            for i, int in enumerate(interaction_vars): 
                r_matrix[i, num_id[int]] = 1

            try: 
                f_test = model_fit.f_test(r_matrix)

                row[f"F_{moderator}"] = f_test.fvalue
                row[f"p_{moderator}"] = f_test.pvalue 
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

def run_lme_fc_model(parcel_df, clinical_df, cluster_df=None, 
                    biotype=False, covariates=[], random_effects="subject", 
                    reml=False, out_df = None):

    parcels = parcel_df.columns.to_list()
    parcels.remove("subject")
    parcels.remove("session")

    data_df = pd.merge(parcel_df, 
                       clinical_df, 
                       on = ["subject", "session"], 
                       how = 'outer')
    
    cluster=False
    clusters=[]
    if cluster_df is not None: 
        data_df = pd.merge(data_df, 
                           cluster_df[["subject", "cluster_label"]], 
                           on = ["subject"])
        cluster=True 

    if DEBUG:
        data_df.to_excel("debug/parcel_fc_merged.xlsx")

    ### code factor levels as dummy variables ###

    # store multi-level factors for ANOVA comparisons
    ml_factors = ["session"]

    # treatment paradigm
    data_df["tx_iTBS"] = data_df["tx_hz"].map(lambda x: 1 if x == 50 else 0)

    if "tx_hz" in covariates: 
        covariates.remove("tx_hz")
        covariates = covariates + ["tx_iTBS"]

    # session 
    data_df["sessionT5"] = data_df["session"].map(lambda x: 1 if x == 1 else 0)
    data_df["sessionEP"] = data_df["session"].map(lambda x: 1 if x == 2 else 0)

    # scan site
    if "site_scan" in covariates:
        data_df["siteMN"] = data_df["site_scan"].map(lambda x: 1 if x == 2000 else 0)
        data_df["sitePV"] = data_df["site_scan"].map(lambda x: 1 if x == 1976 else 0)
        data_df["siteWJ"] = data_df["site_scan"].map(lambda x: 1 if x == 1984 else 0)

        covariates.remove("site_scan")
        covariates = covariates + ["siteMN", "sitePV", "siteWJ"] 

        ml_factors = ml_factors + ["site_scan"]

    # biotype 

    if biotype: 
        data_df["biotypeC+"] = data_df["biotype"].map(lambda x: 1 if x == "C+ biotype" else 0)

        ml_factors = ml_factors + ["biotype"]

    # clusters describing pre-treatment stim site network connectivity
    if cluster:
        num_clusters = len(np.unique(data_df["cluster_label"]))

        for i in range(1, num_clusters):
            data_df[f"cluster{i}"] = data_df["cluster_label"].map(lambda x: 1 if x == i else 0)
            clusters.append(f"cluster{i}")
        
        if num_clusters > 2:
            ml_factors = ml_factors + ["cluster", "cluster_int"]

    print(ml_factors)

    ### specify model regressors ###
    model_params = ["intercept", "sessionT5", "sessionEP"] 

    if biotype: 
        model_params = model_params + ["biotypeC+", "interactionT5xC+", "interactionEPxC+"]

    if cluster: 
        model_params = model_params + clusters

        for ses in ["T5", "EP"]: 
            for i in range(1, num_clusters):
                model_params.append(f"interaction{i}{ses}")

    model_params = model_params + covariates 
    
    num_id = dict(zip(model_params, range(len(model_params)))) 

    if cluster:
        predictor = "cluster x session"
    elif biotype: 
        predictor = "biotype x session"
    else:
        predictor = "session"

    ### set up data frame to store results ### 
    results_df = pd.DataFrame(
        data = {"parcel" : parcels, 
                "predictor" : predictor,
                "outcome" : "parcel_fc",
                "covariates" : ", ".join(covariates)}, 
        columns = ["parcel", "predictor", "outcome", "covariates", "model", 
                   "converged", "aic", "bic"])
        
    results_df.reset_index(drop=True, inplace=True)
    
    for stat in ["coef", "se", "t", "p"]:
        for var in model_params: 
            results_df.insert(len(results_df.columns), f"{stat}_{var}", np.nan)

    for re in [random_effects]: 
        results_df.insert(len(results_df.columns), f"var_{re}", np.nan)
    
    # add test statistics & inferences for multi-level factors & overall model
    for p in ml_factors + ["model"]: 
        results_df.insert(len(results_df.columns), f"F_{p}", np.nan)
        results_df.insert(len(results_df.columns), f"p_{p}", np.nan)

    ### fit each model listed in results data frame and store results ###
    for idx, row in results_df.iterrows():
        parcel = row["parcel"]

        if cluster: 
            formula = f"parcel_fc ~ 1 + cluster x session"
        elif biotype: 
            formula = "parcel_fc ~ 1 + biotype x session"
        else:
            formula = f"parcel_fc ~ 1 + session"

        for cov in covariates: 
            formula = formula + f" + {cov}"

        for re in [random_effects]:
            formula = formula + f" + (1 | {re})"

        row["model"] = formula

        print("Running model: ", formula)
        terms = [parcel, "sessionT5", "sessionEP"]

        if biotype: 
            terms += ["biotypeC+"]

        model_df = data_df[terms + clusters + covariates + [random_effects]]

        # add model intercept
        model_df["intercept"] = 1

        # add biotype interactions
        if biotype: 
            model_df["interactionT5xC+"] = model_df["sessionT5"] * model_df["biotypeC+"]
            model_df["interactionEPxC+"] = model_df["sessionEP"] * model_df["biotypeC+"]

        # add cluster interactions 
        if cluster: 
            for ses in ["T5", "EP"]: 
                for i in range(1, num_clusters):
                    model_df[f"interaction{i}{ses}"] = model_df[f"session{ses}"]*model_df[f"cluster{i}"]

        model_df.dropna(inplace=True)
        model_df.reset_index(inplace=True, drop=True)

        if DEBUG:
            model_df.to_excel("debug/parcel_change_model_data.xlsx",
                              index=False)

        model = sm.MixedLM(model_df[parcel], model_df[model_params], 
                            groups = model_df[random_effects])
        
        model_fit = model.fit(method = ['cg'], reml=reml)
    
        print(model_fit.summary())
        
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
        r_matrix[0, num_id["sessionT5"]] = 1
        r_matrix[1, num_id["sessionEP"]] = 1

        f_test = model_fit.f_test(r_matrix)
        row["F_session"] = f_test.fvalue
        row["p_session"] = f_test.pvalue

        # test significance of scan site if included as a model parameter
        if "site_scan" in ml_factors:
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, num_id["siteMN"]] = 1
            r_matrix[1, num_id["sitePV"]] = 1
            r_matrix[2, num_id["siteWJ"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_site_scan"] = f_test.fvalue
            row["p_site_scan"] = f_test.pvalue

        if biotype: 
            r_matrix = np.zeros(shape = (3, len(model_fit.params)))
            r_matrix[0, num_id["biotypeC+"]] = 1 
            r_matrix[1, num_id["interactionT5xC+"]] = 1
            r_matrix[2, num_id["interactionEPxC+"]] = 1 

            f_test = model_fit.f_test(r_matrix)
            row["F_biotype"] = f_test.fvalue
            row["p_biotype"] = f_test.pvalue

        # test significance of clusters if included as a model parameter
        if cluster:
            r_matrix = np.zeros(shape = (num_clusters-1, len(model_fit.params)))

            for i in range(1, num_clusters):
                r_matrix[i-1, num_id[f"cluster{i}"]] = 1

            f_test = model_fit.f_test(r_matrix)
            row["F_cluster"] = f_test.fvalue
            row["p_cluster"] = f_test.pvalue
        
        # test significance of cluster interactions if included as a model parameter
        if cluster:
            r_matrix = np.zeros(shape = (2*(num_clusters-1), len(model_fit.params)))
            r_idx = 0 

            for ses in ["T5", "EP"]:
                for i in range(1, num_clusters):
                    r_matrix[r_idx, num_id[f"interaction{i}{ses}"]] = 1
                    r_idx += 1

            f_test = model_fit.f_test(r_matrix)
            row["F_cluster_int"] = f_test.fvalue
            row["p_cluster_int"] = f_test.pvalue

        # test significance of overall model 
        r_matrix = np.identity(len(model_fit.params))
        r_matrix = r_matrix[1:]

        f_test = model_fit.f_test(r_matrix)
        row["F_model"] = f_test.fvalue
        row["p_model"] = f_test.pvalue

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
    
    parser.add_argument("clinical_df", 
                        help = "File path to workbook containing all outcome variables & covariates.")
    parser.add_argument("out_fp", help = "Destination file path to store results.")
    parser.add_argument("--cluster_df", help = "Data frame storing a 'cluster_label' " \
    "describing pre-treatment stim network connectivity for each subject.")
    parser.add_argument("--moderator_df", help = "Data frame storing a moderator of FC change")
    parser.add_argument("--atlases", nargs="+", 
                        help = "Workbook(s) containing values of fixed effects variables for atlas parcels.")
    parser.add_argument("--predictor", type=str, help="String description of fixed effects variable.")
    parser.add_argument("--outcomes", nargs="+", 
                         help = "Names of variables in clinical_df to model as outcomes.")
    parser.add_argument("--covariates", nargs="*", 
                        help = "Names of variables in clinical_df to model as covariates.")
    parser.add_argument("--moderator", 
                        help = "Name of variables to model as moderator of network change." \
                        "Only relevant for modeling parcel_fc as outcome.")
    parser.add_argument("--bl_severity", action='store_true', 
                        help="Model baseline symptom severity as a covariate.")
    parser.add_argument("--tx_model", action='store_true', 
                        help = "Explicitly model changes at post-treatment time points.")
    parser.add_argument("--bl_model", action='store_true', 
                                    help="Only uses baseline parcel FC data as a predictor "
                                    "(as opposed to change over treatment)")
    parser.add_argument("--lm", action='store_true', 
                        help="Fit a standard linear regression using one timepoint only.")
    parser.add_argument("--session", type=int, choices=[1,2], help="Model effects for a specific session. " \
    "Must be used with --lm option.")
    parser.add_argument("--fdr_correct", action='store_true', 
                        help = "Correct parcel p-values for multiple comparisons by controlling FDR.")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--biotype", action='store_true')
    parser.add_argument("--include_bl", action='store_true', 
                        help="Whether to include baseline parcel FC as covariate")

    return parser

if __name__ == "__main__": 

    parser = generate_parser()
    args = parser.parse_args()

    print("Atlases: ", args.atlases)

    clinical_df = pd.read_csv(args.clinical_df)
  
    """ 
    # hack for before biotype is included in data-agg
    biotype_df = pd.read_csv("../../data/excel/inputs/biotype.csv")
    biotype_df = biotype_df.loc[biotype_df.session == 0, 
                                ["record_id", "biotype"]]
    
    biotype_df["biotype"] = biotype_df["biotype"].map(lambda x: x.split(" biotype")[0])
    biotype_df["subject"] = biotype_df["record_id"].map(redcap_to_bids)
    biotype_df = biotype_df[["subject", "biotype"]]
    """

    clinical_df["subject"] = clinical_df["record_id"].map(redcap_to_bids)

    parcel_df = combine_atlases(args.atlases)

    sgacc_df = get_bl_stim_sgacc_fc(parcel_df) 
    
    if "parcel_fc" in args.outcomes: 
        if args.cluster_df:
            cluster_df = pd.read_excel(args.cluster_df)
        else:
            cluster_df = None

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
        else: 
            moderator_df = sgacc_df #None
        
        if args.tx_model:
            run_lme_fc_tx_model_2(parcel_df, clinical_df, 
                                covariates = args.covariates,
                                moderator_df = moderator_df,
                                moderator = args.moderator,
                                out_df = args.out_fp, 
                                include_bl = args.include_bl)
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
                                        covariates=args.covariates,
                                        out_df = args.out_fp, 
                                        fdr = args.fdr_correct, 
                                        verbose=args.verbose)
            else:
                if args.bl_model: 
                    results_df = run_bl_lme_tx_model(parcel_df, clinical_df, 
                                            args.outcomes, 
                                            covariates=args.covariates,
                                            out_df = args.out_fp, 
                                            fdr = args.fdr_correct, 
                                            verbose=args.verbose)
                else: 
                    results_df = run_lme_tx_model_2(parcel_df, clinical_df, 
                                            args.outcomes, 
                                            covariates=args.covariates,
                                            out_df = args.out_fp, 
                                            fdr = args.fdr_correct, 
                                            verbose=args.verbose)
        else:
            results_df = run_lme_model(parcel_df, clinical_df, 
                                    "parcel_fc", 
                                    args.outcomes, 
                                    covariates=args.covariates,
                                    out_df = args.out_fp)

    #results_df.to_excel(args.out_fp, index=False)