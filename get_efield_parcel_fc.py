import argparse
import multiprocessing
import nibabel as nib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys 
from timeit import default_timer as timer

sys.path.insert(0, "/oak/stanford/groups/leanew1/users/rdivya/tms/code/utils")
import get_mask 
from gen_seed_FC import gen_seed_roi_FC, gen_seed_voxelwise_FC, gen_seed_ptseries_fc

# PATHS 
DATA_DIR = "/oak/stanford/groups/leanew1/users/rdivya/tms/data"
PLIP_DIR = f"{DATA_DIR}/masks/plip/2021-masks" 
EXCEL_DIR = f"{DATA_DIR}/excel"
SIM_DIR=f"{DATA_DIR}/simulation-outputs"
FC_DIR=f"{DATA_DIR}/efield-fc"
SEED_FC_DIR="/oak/stanford/groups/leanew1/users/rdivya/tms/data/seed-fc"
FPREP_DIR="/oak/stanford/groups/leanew1/SHARED_DATASETS/private/tms/bids\
/data/derivatives/fmriprep-20.2.3/fmriprep"

# Atlases 
ATLAS_DIR="/oak/stanford/groups/leanew1/atlases"
SCHAEFER_DIR="/oak/stanford/groups/leanew1/atlases/schaefer/atlas-Schaefer400"
SCHAEFER_TSV=f"{SCHAEFER_DIR}/\
atlas-Schaefer2018_space-MNI152NLin6Asym_res-2_desc-400Parcels17Networks_dseg.tsv"
TIAN_DIR="/oak/stanford/groups/leanew1/atlases/tian/atlas-TianS2"
TIAN_TSV=f"{TIAN_DIR}/\
atlas-Tian_space-MNI152NLin6Asym_res-2_desc-S23T_dseg.tsv"
YEO_7NETWORKS_DIR = f"{DATA_DIR}/masks/yeo_7networks" 

# filename templates
BASELINE_FC="ses-00_task-rs_acq-mb_space-MNI152NLin6Asym_res-2_desc-denoisedSmoothed"

def load_atlas_tsv(atlas_name, subdir_name): 

    dir_path = Path(f"{ATLAS_DIR}/{subdir_name}/atlas-{atlas_name}")

    tsv_files = list(dir_path.glob("*_dseg.tsv"))

    if len(tsv_files) == 0: 
        sys.stderr.write(f"No atlas found at {dir_path}.\n")
        return
    
    atlas_tsv = pd.read_csv(tsv_files[0], sep='\t')

    return atlas_tsv

def get_stim_brain_voxel_fc(sub, threshold, threshold_type, ses_efield="avg", 
                           ses_fc="00", gm_threshold=0.3, weighted=True,
                           mode="denoisedSmoothed"):
    
    stim_mask_dir = f"{SIM_DIR}/sub-{sub}/ses-{ses_efield}/mni_volumes"
    
    stim_mask_fn = f"sub-{sub}_ses-{ses_efield}_space-MNI152NLin6Asym" \
        f"_res-2_threshold-{threshold}_type-{threshold_type}"
        
    if weighted:
        stim_mask_fn = stim_mask_fn + "_weighted-mask.nii.gz"
    else:
        stim_mask_fn = stim_mask_fn + "_mask.nii.gz"
    
    stim_mask_fp = f"{stim_mask_dir}/{stim_mask_fn}"
    
    if not os.path.isfile(stim_mask_fp):
        sys.stderr.write(f"No E-field mask has been defined for the given threshold for {sub}, {ses_efield}.\n")
        return 
    
    seed_name=f"stim_threshold-{threshold}_type-{threshold_type}"

    if weighted:
        stim_fc_fn = f"sub-{sub}_ses-{ses_fc}_task-rs_acq-mb_space-MNI152NLin6Asym_res-2" \
                    f"_desc-{mode}_seed-{seed_name}_weighted-FC.nii.gz"
    else:
        stim_fc_fn = f"sub-{sub}_ses-{ses_fc}_task-rs_acq-mb_space-MNI152NLin6Asym_res-2" \
                    f"_desc-{mode}_seed-{seed_name}_FC.nii.gz"
        
    gen_seed_voxelwise_FC(sub, ses_fc, seed_name, stim_mask_fp, fc_dir=FC_DIR, 
                      fc_fn = stim_fc_fn, gm_threshold=gm_threshold,
                      mode=mode)

def get_stim_plip_roi_fc(sample_df, threshold, 
                         ses_efield = "avg_coil", ses_fc = "00",
                         weighted = False, out_fp = "stim_plip_roi_fc.xlsx"): 
    
    plip_dir = Path(PLIP_DIR)
    plip_mask_fps = list(plip_dir.glob("*.nii.gz"))

    # load plip ROIs as nibabel images 
    plip_mask_imgs = [nib.load(fp) for fp in plip_mask_fps]

    # set up data frame to store stim-ROI functional connectivity values 
    plip_mask_names = [x.name.split(".nii.gz")[0] for x in plip_mask_fps]

    data_df = pd.DataFrame({"subject" : sample_df.SUBJECT})

    for var in plip_mask_names:
        data_df.insert(len(data_df.columns), var, np.nan)

    efield_mask_fn = f"ses-{ses_efield}_space-MNI152NLin6Asym_res-2" + \
        f"_threshold-{threshold}" 
    
    if weighted: 
        efield_mask_fn += "_weighted-mask.nii.gz"
    else:
        efield_mask_fn += "_mask.nii.gz"
    
    # iterate over participants and quantify stim mask to plip ROI fc
    for idx, row in data_df.iterrows():
        sub = row["subject"]
        print('-'*80 + "\n", sub)

        seed_dir = f"{SIM_DIR}/sub-{sub}/ses-{ses_efield}/mni_volumes"
        seed_fp = f"{seed_dir}/sub-{sub}_{efield_mask_fn}" 

        seed_img = nib.load(seed_fp)

        start = timer()
        row[1:] = gen_seed_roi_FC(sub, ses_fc, seed_img, plip_mask_imgs, 
                                  seed_intersect_gm = True, 
                                  roi_intersect_gm = True, 
                                  gm_threshold = 0.25)
        end = timer()

        print(f"Time to extract FCs for {sub}: {end-start} s\n")
        data_df.iloc[idx] = row

    if out_fp is not None:
        data_df.to_excel(out_fp, index = False)
    
    return data_df 

def get_stim_yeo_roi_fc(sample_df, threshold, network = "all",
                         ses_efield = "avg_coil", ses_fc = "00",
                         weighted = False, out_fp = "stim_plip_roi_fc.xlsx"): 
    
    yeo_dir = Path(YEO_7NETWORKS_DIR)

    if network == "all":
        yeo_mask_fps = list(yeo_dir.glob("*.nii.gz"))
    else:
        yeo_mask_fps = list(yeo_dir.glob(f"*{network}*.nii.gz"))

    # load plip ROIs as nibabel images 
    yeo_mask_imgs = [nib.load(fp) for fp in yeo_mask_fps]

    # set up data frame to store stim-ROI functional connectivity values 
    yeo_mask_names = [x.name.split("_space")[0] for x in yeo_mask_fps]

    data_df = pd.DataFrame({"subject" : sample_df.SUBJECT})

    for var in yeo_mask_names:
        data_df.insert(len(data_df.columns), var, np.nan)

    efield_mask_fn = f"ses-{ses_efield}_space-MNI152NLin6Asym_res-2" + \
        f"_threshold-{threshold}" 
    
    if weighted: 
        efield_mask_fn += "_weighted-mask.nii.gz"
    else:
        efield_mask_fn += "_mask.nii.gz"
    
    # iterate over participants and quantify stim mask to plip ROI fc
    for idx, row in data_df.iterrows():
        sub = row["subject"]
        print('-'*80 + "\n", sub)

        seed_dir = f"{SIM_DIR}/sub-{sub}/ses-{ses_efield}/mni_volumes"
        seed_fp = f"{seed_dir}/sub-{sub}_{efield_mask_fn}" 

        seed_img = nib.load(seed_fp)

        start = timer()
        row[1:] = gen_seed_roi_FC(sub, ses_fc, seed_img, yeo_mask_imgs, 
                                  seed_intersect_gm = True, 
                                  roi_intersect_gm = True, 
                                  gm_threshold = 0.25)
        end = timer()

        print(f"Time to extract FCs for {sub}: {end-start} s")
        data_df.iloc[idx] = row

    if out_fp is not None:
        data_df.to_excel(out_fp, index = False)
    
    return data_df 

def get_stim_mask_parcel_fc(sub, ses, threshold, ses_efield, weighted, 
                            nparcels, seg_name): 
    
    # set up stimulation mask file name
    efield_mask_fn = f"sub-{sub}_ses-{ses_efield}_space-MNI152NLin6Asym_res-2" + \
        f"_threshold-{threshold}" 
    
    if weighted: 
        efield_mask_fn += "_weighted-mask.nii.gz"
    else:
        efield_mask_fn += "_mask.nii.gz"

    seed_fp = f"{SIM_DIR}/sub-{sub}/ses-{ses_efield}/mni_volumes/{efield_mask_fn}" 

    if not os.path.isfile(seed_fp):
        print(f"Could not find stimulation mask for {sub} at {seed_fp}")
        return np.full(nparcels, np.nan)
    
    seed_img = nib.load(seed_fp)

    start = timer()
    parcel_fcs = gen_seed_ptseries_fc(sub, ses, seed_img, seg_name, 
                                        nparcels)
    end = timer()
    print(f"Time to extract FCs for {sub}: {end-start} s")

    print("Parcel FCs: ", parcel_fcs)
    
    return parcel_fcs

def get_efield_cog_parcel_fc(sub, ses, efield_cog, nparcels, seg_name): 
    ref_fp = f"{FPREP_DIR}/sub-{sub}/anat/sub-{sub}_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz"

    if np.any(np.isnan(efield_cog)): 
        return np.full(nparcels, np.nan)
    
    if not os.path.isfile(ref_fp):
        ref_fp = f"{FPREP_DIR}/sub-{sub}/ses-00/anat/sub-{sub}_ses-00_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz"

        if not os.path.isfile(ref_fp): 
            print(f"Could not find reference T1 in MNI space for {sub} at {ref_fp}.")
            return np.full(nparcels, np.nan)

    seed_img = get_mask.get_spherical_roi_mask(ref_fp, efield_cog, 10)

    start = timer()
    parcel_fcs = gen_seed_ptseries_fc(sub, ses, seed_img, seg_name, 
                                        nparcels)
    end = timer()
    print(f"Time to extract FCs for {sub}: {end-start} s")

    return parcel_fcs 

def get_efield_parcel_fc_group(sample_df, seg_dir, seg_name, nthreads, 
                        efield_cog_df = None, threshold = None, 
                        ses_efield = "avg_coil", weighted = True, 
                        ses_list = ["00", "01", "02"], 
                        out_fp = "efield_cog_schaefer_parcel_fc.xlsx"):
    
    # load in atlas 
    atlas = load_atlas_tsv(seg_name, seg_dir)
    labels = list(atlas["label"])
    nparcels = len(labels)

    subjects = sample_df.SUBJECT

    # set up data frame 
    fc_df = pd.DataFrame({"subject" : np.repeat(subjects, len(ses_list)), 
                          "session" : np.tile(ses_list, len(subjects))})
    
    fc_df = fc_df.reset_index(drop=True)
    
    # average available sessions for each subject & bind to fc_df
    if efield_cog_df is not None: 
        efield_cog_df = efield_cog_df.groupby(["SUBJECT"]).mean().reset_index()
        efield_cog_df = efield_cog_df[["SUBJECT", 
                                       "X_CTR_MNI", "Y_CTR_MNI", "Z_CTR_MNI"]]
        
        efield_cog_df.rename(columns = lambda x : x.lower(), inplace = True)
        
        fc_df = pd.merge(fc_df, efield_cog_df, how="left", on = "subject")

    nrows = len(fc_df)

    # set up multithreading
    pool = multiprocessing.Pool(nthreads)

    # call appropriate function based on seed type
    if efield_cog_df is not None:
        fcs = pool.starmap(get_efield_cog_parcel_fc, zip(
            fc_df["subject"].to_numpy(), 
            fc_df["session"].to_numpy(), 
            fc_df[["x_ctr_mni", "y_ctr_mni", "z_ctr_mni"]].to_numpy(), 
            np.repeat(nparcels, nrows), 
            np.repeat(seg_name, nrows)
        )) 

    elif threshold is not None: 
        fcs = pool.starmap(get_stim_mask_parcel_fc, zip(
          fc_df["subject"].to_numpy(), 
          fc_df["session"].to_numpy(), 
          np.repeat(threshold, nrows), 
          np.repeat(ses_efield, nrows), 
          np.repeat(weighted, nrows), 
          np.repeat(nparcels, nrows), 
          np.repeat(seg_name, nrows)
        ))

    fcs = np.array(fcs)

    for col, label in enumerate(labels):
        fc_df[label] = fcs[:, col]

    if out_fp is not None:
        print(f"Saving results to {out_fp}...")
        fc_df.to_excel(out_fp, index = False)
    
    return fc_df 
        
def get_stim_seed_fc_spatial_corr(sample_df, stim_threshold, seeds,
                                ses_efield="avg_coil", 
                                out_fp = "stim_seed_fc_spatial_corr.xlsx"):
    
    spatial_corrs = pd.DataFrame({"subject" : sample_df.SUBJECT})

    for seed in seeds: 
        spatial_corrs.insert(len(spatial_corrs.columns), seed, np.nan)

    for idx, row in spatial_corrs.iterrows():
        sub = row["subject"]
        print(sub)
        # load subject E-field stimulation mask  
        efield_dir = f"{SIM_DIR}/sub-{sub}/ses-{ses_efield}/mni_volumes"
        stim_mask_fp = f"{efield_dir}/sub-{sub}_ses-{ses_efield}" + \
            f"_space-MNI152NLin6Asym_res-2_threshold-{stim_threshold}" + \
            "_weighted-mask.nii.gz"
        
        if not os.path.isfile(stim_mask_fp):
            print(f"No stimulation mask found for {sub} at {stim_mask_fp}")
            continue
        
        stim_mask_img = nib.load(stim_mask_fp)
        stim_mask = np.asarray(stim_mask_img.get_fdata())
        stim_roi_idx = np.flatnonzero(stim_mask)

        for seed in seeds:
            print(seed)
            # load subject baseline seed FC volume    
            seed_fc_fp = f"{SEED_FC_DIR}/sub-{sub}/ses-00/" + \
                f"sub-{sub}_{BASELINE_FC}_seed-{seed}_FC.nii.gz"
            
            if not os.path.isfile(seed_fc_fp):
                print(f"No {seed} FC volume found for {sub} at {seed_fc_fp}")
                continue
            
            seed_fc_img = nib.load(seed_fc_fp)
            assert(np.all(seed_fc_img.affine == stim_mask_img.affine))

            seed_fc = np.asarray(seed_fc_img.get_fdata())
            
            # find voxels where seed FC is not NaN 
            # i.e., within the freesurfer brain mask
            seed_fc_idx = np.flatnonzero(~np.isnan(seed_fc))

            roi_idx = np.intersect1d(stim_roi_idx, seed_fc_idx)

            spatial_corr = np.corrcoef(stim_mask.take(roi_idx), 
                                    seed_fc.take(roi_idx))[0,1]
            
            row[seed] = spatial_corr

        spatial_corrs.iloc[idx] = row
    
    spatial_corrs.to_excel(out_fp, index=False)

    return spatial_corrs

def generate_parser():
    parser = argparse.ArgumentParser(prog="Analyze_EField", 
            description="Characterize stimulated region in subject and MNI space from E-field simulation results.")
    
    parser.add_argument("sample_fp", help="Excel workbook containing subject IDs to process.")
    parser.add_argument("out_fp", help="File path to save results.")
    parser.add_argument("atlas_name")
    parser.add_argument("atlas_dir")
    parser.add_argument("--ses_list", nargs='*', default=["00", "01", "02"])
    parser.add_argument("--efield_cog_fp", help="Excel workbook containing E-field center of gravity.")
    parser.add_argument("--threshold", help="E-field threshold to define stimulation mask; e.g. 95.0perc.")
    parser.add_argument("--weighted", action='store_true')
    parser.add_argument("--nthreads", type=int, default=1, help="Number of threads to use for parallelization.")
    parser.add_argument("--ses_efield", default="avg_coil", help="Session label for E-field simulation.")

    return parser
    
if __name__ == "__main__":

    parser = generate_parser()
    args = parser.parse_args()
    
    sample_df = pd.read_excel(args.sample_fp)

    if args.efield_cog_fp is not None:
        efield_cog_df = pd.read_excel(args.efield_cog_fp)

        get_efield_parcel_fc_group(sample_df, args.atlas_dir, args.atlas_name, 
            args.nthreads, efield_cog_df = efield_cog_df, ses_list = args.ses_list, 
            out_fp = args.out_fp)
    
    if args.threshold is not None:
        get_efield_parcel_fc_group(sample_df, args.atlas_dir, args.atlas_name, 
            args.nthreads, threshold = args.threshold, ses_efield = args.ses_efield, 
            weighted = args.weighted, ses_list = args.ses_list, 
            out_fp = args.out_fp)