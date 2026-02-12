import ants
import nibabel as nib
import numpy as np
import os
import pandas as pd 
from pathlib import Path
import sys

ATLAS_DIR="/oak/stanford/groups/leanew1/atlases"
DATA_DIR="/oak/stanford/groups/leanew1/users/rdivya/simnibs/data"
EXCEL_DIR=f"{DATA_DIR}/excel"
SIM_DIR=f"{DATA_DIR}/simulation-outputs"
FPREP_DIR="/oak/stanford/groups/leanew1/SHARED_DATASETS/private/tms/bids/data/derivatives/fmriprep-20.2.3/fmriprep"

def get_efield_mni_arr(sub, ses="avg_coil"): 
    sim_dir = Path(f"{SIM_DIR}/sub-{sub}/ses-{ses}")

    efield_files = list(sim_dir.glob("mni_volumes/*_magnE.nii.gz"))

    if len(efield_files) == 0:
        sys.stderr.write(f"No E-field simulation found for {sub}, ses-{ses}.\n")
        return 
    
    efield_img = nib.load(efield_files[0])
    efield_arr = np.array(efield_img.get_fdata())

    return efield_arr 

def load_atlas(dir, name):
    atlas_dir = Path(f"{ATLAS_DIR}/{dir}/atlas-{name}")

    seg_files = list(atlas_dir.glob("*_dseg.nii*"))

    if len(seg_files) == 0: 
        sys.stderr.write(f"No atlas found in {atlas_dir}.\n")
        return 
    
    seg = nib.load(seg_files[0])
    affine = seg.affine

    if not np.all(np.diag(affine) > 0):
        seg_img = ants.image_read(seg_files[0].as_posix(), 
                                  reorient = 'LPI')
        
        print("New image orientation: ", seg_img.get_orientation())
        
        seg_arr = seg_img.numpy()
    else:
        seg_arr = np.array(seg.get_fdata())

    tsv_files = list(atlas_dir.glob("*_dseg.tsv"))
    tsv = pd.read_csv(tsv_files[0], sep='\t')

    return seg_arr, tsv 

def get_gm_seg(sub, space="MNI152NLin6Asym", res = "2"):

    gm_fp = f"{FPREP_DIR}/sub-{sub}/anat/sub-{sub}_space-{space}_res-{res}_label-GM_probseg.nii.gz"

    if not os.path.isfile(gm_fp):
        gm_fp = f"{FPREP_DIR}/sub-{sub}/ses-00/anat/sub-{sub}_ses-00_space-{space}_res-{res}_label-GM_probseg.nii.gz"
        
        if not os.path.isfile(gm_fp):
            return None
        
    gm_img = nib.load(gm_fp)
    gm_arr = np.array(gm_img.get_fdata())

    return gm_arr
    
def get_efield_parcel(subjects, seg_dir, seg_name,
                      ses="avg_coil", gm_thresh=0.5):
    
    atlas = load_atlas(seg_dir, seg_name)

    if atlas is None:
        print("Could not load atlas!")
        return 
    
    seg_arr, parcels = atlas
    indices = list(parcels["index"])
    labels = list(parcels["label"])

    efield_df = pd.DataFrame(data = 
                {"subject" : subjects, 
                "session" : np.repeat(ses, len(subjects))}, 
                columns = ["subject", "session"] + labels)
    
    efield_df.reset_index(drop=True, inplace=True)
    
    for idx, row in efield_df.iterrows():
        sub = row["subject"] 
        efield_arr = get_efield_mni_arr(sub)
        gm_arr = get_gm_seg(sub)

        if efield_arr is None: 
            sys.stderr.write(f"No E-field data found for {sub}, {ses}.\n")
            continue 

        if gm_thresh > 0:
            if gm_arr is None : 
                sys.stderr.write(f"No gray matter segmentation found for {sub}.\n")
                continue  

            efield_arr[gm_arr < gm_thresh] = np.nan
        
        parcel_efield = [np.nanmean(efield_arr[seg_arr == idx]) for idx in indices]

        row[2:] = parcel_efield

        efield_df.iloc[idx] = row

    return efield_df

if __name__ == "__main__":

    sample_df = pd.read_excel(sys.argv[1])
    subjects = sample_df.SUBJECT

    schaefer_df = get_efield_parcel(subjects, "schaefer", "Schaefer400")
    schaefer_df.to_excel(sys.argv[1],
                         index=False)
    
    tian_df = get_efield_parcel(subjects, "tian", "TianS2")
    tian_df.to_excel(sys.argv[2])