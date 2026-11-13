import argparse
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps as cm
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np 
import os
import pandas as pd
import sys

TMS_DIR="/oak/stanford/groups/leanew1/SHARED_DATASETS/private/tms/bids/data"
FS_DIR = f"{TMS_DIR}/derivatives/fmriprep-20.2.3/freesurfer"

DATA_DIR="/oak/stanford/groups/leanew1/users/rdivya/tms/data"
ATLAS_DIR=f"{DATA_DIR}/atlases"
SCHAEFER_ATLAS_SURF=f"{ATLAS_DIR}/Schaefer/fsaverage/lh.Schaefer2018_400Parcels_17Networks_order.annot"

my_cmap1 = colors.LinearSegmentedColormap.from_list('my_gradient', (
                 # Edit this gradient at https://eltos.github.io/gradient/#002610-05CC67-FFFFFF-D5007A-700E4F
                 (0.000, (0.000, 0.149, 0.063)),
                 (0.250, (0.020, 0.800, 0.404)),
                 (0.500, (1.000, 1.000, 1.000)),
                 (0.750, (0.835, 0.000, 0.478)),
                 (1.000, (0.439, 0.055, 0.310))))

def plot_cluster_effects(cluster_df, results_df, outcome, effect="parcel", 
                         stat="psize", vars=[], alpha=0.05, 
                         flip_valence=False, see_all = False, svg=False, 
                         views = {"lh": {"lateral" : [0, 180], "medial" : [0, 0]},
                                 "rh": {"lateral" : [0, 0], "medial" : [0, 180]}}, 
                        t_max=None, cmap='RdBu_r', plot_boundaries=False): 

    # extract results associated with outcome 
    results_df = results_df[results_df.outcome == outcome]    
    results_df.reset_index(drop=True, inplace=True)

    cluster_df = cluster_df[cluster_df.outcome == outcome]
    cluster_df.reset_index(drop=True, inplace=True)

    # hack to visualize all parcel t-stats
    if see_all:
        cluster_df[f"p_perm_{stat}"] = 0 

    # specify direction of change using coefficients specified in 'direction'
    # useful when plotting F-statistic 
    if len(vars) > 0:
        coefs = np.sum([results_df.loc[0:400, f"coef_{v}"] for v in vars],
                        axis=0) 

        #coefs[np.abs(coefs) < tol] = 0

        # shift values to the right to account for 1-indexing of Schaefer atlas
        np.insert(coefs[0:400], 0, 0)

    # flip scale valence if specified 
    if flip_valence: 
        cluster_df[f"cluster_{stat}"] = -1 * cluster_df[f"cluster_{stat}"]

    # set up colormap to normalize over range of observed test statistics 
    cmap = cm[cmap]

    if t_max is None:
        t_max = np.max(np.abs(cluster_df[f"cluster_{stat}"]))

    norm = colors.Normalize(vmin = -t_max, 
                            vmax = t_max)

    for hemi in views.keys():
        ax = plt.figure().add_subplot(projection='3d')
        plt.tight_layout()

        # load schaefer atlas 
        atlas_fp =f"{ATLAS_DIR}/Schaefer/fsaverage/{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"
        labels, _, names = nib.freesurfer.io.read_annot(atlas_fp)

        names = [str(name).strip("b'") for name in names]

        # load fsaverage template 
        template_fp = f"{FS_DIR}/fsaverage/surf/{hemi}.pial_semi_inflated"
        template = nib.freesurfer.io.read_geometry(template_fp)
        coords, triangles = template[0:2]

        # assign a label to each triangle based on 'median' of vertex labels 
        label_tri = np.array([np.percentile(labels[tri], 50, 
                                            method='closest_observation') 
                            for tri in triangles], dtype=int)

        if hemi == "rh":
            label_tri[label_tri > 0] += 200 

        # assign a p-value to each triangle based on its membership to a cluster
        p_tri = np.full(len(label_tri), 1.0, dtype=float)
        t_tri = np.full(len(label_tri), 0.0)

        for _, cluster in cluster_df.iterrows():
            if cluster[f"p_perm_{stat}"] > alpha:
                continue 

            parcels = cluster["parcel_ids"].strip('[]')
            parcels = np.fromstring(parcels, sep=' ', dtype=int)
            print("Cluster: ", parcels)

            factor = 1
            
            if len(vars) > 0:
                net_coef = np.sum(coefs[parcels])

                if net_coef < 0:
                    factor = -1
            
            print("Factor: ", factor)

            in_cluster = np.isin(label_tri, parcels)
            print("Num. parcels: ", len(parcels))
            print("Num. triangles: ", np.sum(in_cluster))
            p_tri[in_cluster] = cluster[f"p_perm_{stat}"]
            t_tri[in_cluster] = factor*cluster[f"cluster_{stat}"]

        color_tri = cmap(norm(t_tri))
        
        # mask out clusters that do not meet significance threshold
        # based on permuted null distribution
        color_tri[p_tri >= alpha] = colors.to_rgba('lightgrey')

        # extract parcel boundaries and color black 
        if plot_boundaries:
            bdry_tri = np.array([not np.all(labels[tri] == labels[tri[0]]) 
                                for tri in triangles]).astype(bool)
                
            color_tri[bdry_tri] = colors.to_rgba('dimgrey')

        # set up axis views for plotting 

        # plot fsaverage template with surface patches colored according to 
        # the Yeo 7-Network parcellation
        ax.axis("off")
        ax.margins(0,0,0)
        ax.set_aspect('equal', adjustable='box')

        surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], 
                        triangles = triangles, 
                        alpha=1, linewidth=0, zorder=1,
                        antialiased=False, shade=True)
        surf.set_fc(color_tri)

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, 
                    shrink = 0.4, 
                    orientation = 'horizontal', 
                    label = stat, pad = -.15)

        for view, angles in views[hemi].items():
            elev, azim = angles
            ax.view_init(elev=elev, azim=azim) # azim=135

            fig_dir = f"figs/clusters/transl_psych_revision/{outcome}/"
            os.makedirs(fig_dir, exist_ok=True)

            fn = f"{outcome}_vs_{effect}_cluster-{stat}_schaefer400_{hemi}_{view}"

            if svg: 
                plt.savefig(f"{fig_dir}/{fn}.svg", bbox_inches = 'tight')
            else: 
                plt.savefig(f"{fig_dir}/{fn}.png", bbox_inches = 'tight', dpi=600)

def plot_effects(results_df, outcome, effect="parcel", F_stat=False, 
                 corrected=False, vars=[], alpha=0.05, flip_valence=False,
                 cmap='RdBu_r', see_all=False, plot_boundaries=False, t_max=None,
                 views = {"lh": {"lateral" : [0, 180], "medial" : [0, 0]},
                                 "rh": {"lateral" : [0, 0], "medial" : [0, 180]}}): 

    # add custom dorsolateral prefrontal view angle 
    #views["lh"]["dorsolateral"] = [10, 170]

    # extract results associated with outcome 
    results_df = results_df[results_df.outcome == outcome]    
    results_df.reset_index(drop=True, inplace=True)

    # extract p-values and t-values for parcel coefficient
    if see_all:
        p_values = np.zeros(400)
    elif corrected:
        p_values = results_df[f"p_{effect}_corr"].to_numpy()
    else:
        p_values = results_df[f"p_{effect}"].to_numpy()

    if F_stat:
        t_values = results_df[f"F_{effect}"].to_numpy()
    else:
        t_values = results_df[f"t_{effect}"].to_numpy()

    # flip scale valence if specified 
    if flip_valence: 
        t_values = -1 * t_values

    # specify direction of change using coefficients specified in 'direction'
    # useful when plotting F-statistic 
    if F_stat and len(vars) > 0:
        coefs = np.sum([results_df.loc[0:400, f"coef_{v}"] for v in vars],
                        axis=0) 
        np.insert(coefs[0:400], 0, 0)

        # change direction of test statistic to reflect direction of effect 
        t_values[coefs < 0] = -1 * t_values[coefs < 0]

    # set up colormap to normalize over range of observed test statistics 
    cmap = cm[cmap]

    if t_max is None:
        t_max = np.nanmax(np.abs(t_values))

    norm = colors.Normalize(vmin = -t_max, 
                            vmax = t_max)
    
    for hemi in views.keys():
        ax = plt.figure().add_subplot(projection='3d')
        plt.tight_layout()

        # load schaefer atlas 
        atlas_fp =f"{ATLAS_DIR}/Schaefer/fsaverage/{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"
        labels, _, _ = nib.freesurfer.io.read_annot(atlas_fp)

        # load fsaverage template 
        template_fp = f"{FS_DIR}/fsaverage/surf/{hemi}.pial_semi_inflated"
        template = nib.freesurfer.io.read_geometry(template_fp)
        coords, triangles = template[0:2]

        # assign a label to each triangle based on 'median' of vertex labels 
        label_tri = np.array([np.percentile(labels[tri], 50, 
                                            method='closest_observation') 
                            for tri in triangles], dtype=int)

        # extract relevant significance values 
        # only keep values corresponding to Schaefer atlas (i.e., discard Tian)
        if hemi == "lh":
            p_values_hemi = p_values[0:200]
            t_values_hemi = t_values[0:200]
        else: 
            p_values_hemi = p_values[200:400]
            t_values_hemi = t_values[200:400]

        # shift values to the right to account for 1-indexing of Schaefer atlas
        # in both hemispheres
        p_values_hemi = np.insert(p_values_hemi, 0, 1)
        t_values_hemi = np.insert(t_values_hemi, 0, 0) 

        p_tri = p_values_hemi[label_tri]
        t_tri = t_values_hemi[label_tri]

        color_tri = cmap(norm(t_tri))
        
        # mask out non-significant parcels
        grey = colors.to_rgba('lightgrey') # 'snow'
        color_tri[p_tri >= alpha] = grey

        # extract parcel boundaries & color black 

        if plot_boundaries:
            bdry_tri = np.array([not np.all(labels[tri] == labels[tri[0]]) 
                                for tri in triangles]).astype(bool)            
            color_tri[bdry_tri] = colors.to_rgba('dimgrey')

        # set up axis views for plotting 

        # plot fsaverage template with surface patches colored according to 
        # the Yeo 7-Network parcellation
        ax.axis("off")
        ax.margins(0,0,0)
        ax.set_aspect('equal', adjustable='box')

        surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], 
                        triangles = triangles, 
                        alpha=1, linewidth=0, zorder=1,
                        antialiased=False, shade=True)
        surf.set_fc(color_tri)

        if F_stat: 
            label = "$F$"
        else:
            label = "$t$"
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, 
                    shrink = 0.4, 
                    orientation = 'horizontal', 
                    label = label, pad = -.15)

        cbar.ax.set_xticks(np.linspace(-t_max, t_max, 5))

        for view, angles in views[hemi].items():
            elev, azim = angles
            ax.view_init(elev=elev, azim=azim) # azim=135
            
            fig_dir = f"figs/parcels/predictor-ParcelFC/all/{outcome}/"
            os.makedirs(fig_dir, exist_ok=True)

            fn = f"{outcome}_vs_{effect}_schaefer400_{hemi}_{view}.png"
            plt.savefig(f"{fig_dir}/{fn}", bbox_inches = 'tight', dpi=600)

def plot_fc(fc_df, sub, ses, cmap='RdBu_r',
            views = {"lh": {"lateral" : [0, 180], 
                            "medial" : [0, 0]},
                    "rh": {"lateral" : [0, 0], 
                         "medial" : [0, 180]}}): 

    # extract results associated with outcome 
    sub_df = fc_df[fc_df.subject == sub]
    ses_df = sub_df[sub_df.session == ses]

    parcel_fcs = ses_df.T.reset_index(drop=True)
    parcel_fcs = parcel_fcs.iloc[2:, 0].to_numpy(dtype=float)
    parcel_fcs = np.nan_to_num(parcel_fcs)

    print(len(parcel_fcs))
    
    # set up colormap to normalize over range of observed test statistics 
    cmap = cm[cmap]
    fc_max = np.max(np.abs(parcel_fcs))

    norm = colors.Normalize(vmin = -fc_max, 
                            vmax = fc_max)
    
    for hemi in views.keys():
        ax = plt.figure().add_subplot(projection='3d')
        plt.tight_layout()

        # load schaefer atlas 
        atlas_fp =f"{ATLAS_DIR}/Schaefer/fsaverage/{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"
        labels, color_lut, _ = nib.freesurfer.io.read_annot(atlas_fp)

        # load fsaverage template 
        template_fp = f"{FS_DIR}/fsaverage/surf/{hemi}.pial_semi_inflated"
        template = nib.freesurfer.io.read_geometry(template_fp)
        coords, triangles = template[0:2]

        # assign a label to each triangle based on 'median' of vertex labels 
        label_tri = np.array([np.percentile(labels[tri], 50, 
                                            method='closest_observation') 
                            for tri in triangles], dtype=int)


        if hemi == "lh": 
            hemi_fcs = np.insert(parcel_fcs[0:200], 0, 0)
        if hemi == "rh": 
            hemi_fcs = np.insert(parcel_fcs[200:400], 0, 0)

        fc_tri = hemi_fcs[label_tri]

        color_tri = cmap(norm(fc_tri))

        # extract parcel boundaries & color black 
        bdry_tri = np.array([not np.all(labels[tri] == labels[tri[0]]) 
                            for tri in triangles]).astype(bool)            
        color_tri[bdry_tri] = [0, 0, 0, 0.5]

        # set up axis views for plotting 

        # plot fsaverage template with surface patches colored according to 
        # the Yeo 7-Network parcellation
        ax.axis("off")
        ax.margins(0,0,0)
        ax.set_aspect('equal', adjustable='box')

        surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], 
                        triangles = triangles, 
                        alpha=1, linewidth=0, zorder=1,
                        antialiased=False, shade=True)
        surf.set_fc(color_tri)

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, 
                    shrink = 0.4, 
                    orientation = 'horizontal', 
                    label = "FC", pad = -.15)

        for view, angles in views[hemi].items():
            elev, azim = angles
            ax.view_init(elev=elev, azim=azim) # azim=135
            
            fig_dir = f"figs/parcels/fc/sub-{sub}"
            os.makedirs(fig_dir, exist_ok=True)

            fn = f"sub-{sub}_ses-{ses}_fc_schaefer400_{hemi}_{view}.png"
            plt.savefig(f"{fig_dir}/{fn}", bbox_inches = 'tight', dpi=600)

def plot_coefs(results_df, outcome, vars, label, factors=None, 
            mask_vars=[], mask=False, alpha=0.05, c_max=None, cmap='RdBu_r',
            views = {"lh": {"lateral" : [0, 180], "medial" : [0, 0]},
                    "rh": {"lateral" : [0, 0], "medial" : [0, 180]}}): 

    # extract results associated with outcome 
    results_df = results_df[results_df.outcome == outcome]    
    results_df.reset_index(drop=True, inplace=True)

    coefs = [f"coef_{var}" for var in vars]
    coefs = results_df.loc[0:400, coefs].to_numpy()

    if mask:
        if len(mask_vars) > 0: 
            p_names = [f"p_{var}" for var in mask_vars]
            p_values = results_df.loc[0:400, p_names].to_numpy()
            p_values = np.min(p_values, axis=1)
        else: 
            p_values = results_df.loc[0:400, "p_model"].to_numpy()

    if factors is None: 
        coefs = np.sum(coefs, axis=1)
    else: 
        coefs = np.sum(coefs * factors, axis=1)

    # set up colormap to normalize over range of observed test statistics 
    cmap = cm[cmap]

    if c_max is None:
        c_max = np.max(np.abs(coefs))

    norm = colors.Normalize(vmin = -c_max, 
                            vmax = c_max)
    
    for hemi in views.keys():
        ax = plt.figure().add_subplot(projection='3d')
        plt.tight_layout()

        # load schaefer atlas 
        atlas_fp =f"{ATLAS_DIR}/Schaefer/fsaverage/{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"
        labels, color_lut, _ = nib.freesurfer.io.read_annot(atlas_fp)

        # load fsaverage template 
        template_fp = f"{FS_DIR}/fsaverage/surf/{hemi}.pial_semi_inflated"
        template = nib.freesurfer.io.read_geometry(template_fp)
        coords, triangles = template[0:2]

        # assign a label to each triangle based on 'median' of vertex labels 
        label_tri = np.array([np.percentile(labels[tri], 50, 
                                            method='closest_observation') 
                            for tri in triangles], dtype=int)

        # extract relevant significance values 
        # only keep values corresponding to Schaefer atlas (i.e., discard Tian)
        if hemi == "lh":
            coefs_hemi = coefs[0:200]
        else: 
            coefs_hemi = coefs[200:400]

        # shift values to the right to account for 1-indexing of Schaefer atlas
        # in both hemispheres
        coefs_hemi = np.insert(coefs_hemi, 0, 0)

        if mask: 
            if hemi == "lh":
                p_hemi = p_values[0:200]
            else: 
                p_hemi = p_values[200:400]
            
            p_hemi = np.insert(p_hemi, 0, 1)

        coef_tri = coefs_hemi[label_tri]
        color_tri = cmap(norm(coef_tri))

        if mask: 
            p_tri = p_hemi[label_tri]
            print(np.min(p_tri))
            print(np.max(p_tri))
            color_tri[p_tri > alpha] = colors.to_rgba('lightgrey')
        
        # extract parcel boundaries & color black 
        bdry_tri = np.array([not np.all(labels[tri] == labels[tri[0]]) 
                            for tri in triangles]).astype(bool)            
        color_tri[bdry_tri] = colors.to_rgba('dimgrey')

        # set up axis views for plotting 

        # plot fsaverage template with surface patches colored according to 
        # the Yeo 7-Network parcellation
        ax.axis("off")
        ax.margins(0,0,0)
        ax.set_aspect('equal', adjustable='box')

        surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], 
                        triangles = triangles, 
                        alpha=1, linewidth=0, zorder=1,
                        antialiased=False, shade=True)
        surf.set_fc(color_tri)

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, 
                    shrink = 0.4, 
                    orientation = 'horizontal', 
                    pad = -.15)

        for view, angles in views[hemi].items():
            elev, azim = angles
            ax.view_init(elev=elev, azim=azim) # azim=135
            
            fig_dir = f"figs/coefs/{outcome}/"
            os.makedirs(fig_dir, exist_ok=True)

            fn = f"{outcome}_predict-{label}_schaefer400_{hemi}_{view}.png"
            plt.savefig(f"{fig_dir}/{fn}", bbox_inches = 'tight', dpi=600)

def plot_mask(parcels, mask_name, mask_color='honeydew',
            views = {"lh": {"lateral" : [0, 180], 
                            "medial" : [0, 0]},
                    "rh": {"lateral" : [0, 0], 
                         "medial" : [0, 180]}}): 

    for hemi in views.keys():
        ax = plt.figure().add_subplot(projection='3d')
        plt.tight_layout()

        # load schaefer atlas 
        atlas_fp =f"{ATLAS_DIR}/Schaefer/fsaverage/{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"
        labels, color_lut, _ = nib.freesurfer.io.read_annot(atlas_fp)

        # load fsaverage template 
        template_fp = f"{FS_DIR}/fsaverage/surf/{hemi}.pial_semi_inflated"
        template = nib.freesurfer.io.read_geometry(template_fp)
        coords, triangles = template[0:2]

        # assign a label to each triangle based on 'median' of vertex labels 
        label_tri = np.array([np.percentile(labels[tri], 50, 
                                            method='closest_observation') 
                            for tri in triangles], dtype=int)

        if hemi == "rh":
                    label_tri += 200 
            
        color_tri = np.full((len(label_tri), 4), 
                            colors.to_rgba('lightgrey'))
        
        mask_tri = [t in parcels for t in label_tri]
        color_tri[mask_tri] = colors.to_rgba(mask_color)

        # extract parcel boundaries & color black 
        bdry_tri = np.array([not np.all(labels[tri] == labels[tri[0]]) 
                            for tri in triangles]).astype(bool)            
        color_tri[bdry_tri] = colors.to_rgba('dimgrey')

        # set up axis views for plotting 

        # plot fsaverage template with surface patches colored according to 
        # the Yeo 7-Network parcellation
        ax.axis("off")
        ax.margins(0,0,0)
        ax.set_aspect('equal', adjustable='box')

        surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], 
                        triangles = triangles, 
                        alpha=1, linewidth=0, zorder=1,
                        antialiased=False, shade=True)
        surf.set_fc(color_tri)

        for view, angles in views[hemi].items():
            elev, azim = angles
            ax.view_init(elev=elev, azim=azim) # azim=135
            
            fig_dir = f"figs/masks/"
            os.makedirs(fig_dir, exist_ok=True)

            fn = f"{mask_name}_schaefer400_{hemi}_{view}.png"
            plt.savefig(f"{fig_dir}/{fn}", bbox_inches = 'tight', dpi=600)


def generate_parser():
    parser = argparse.ArgumentParser(prog="plot_efield_effects", 
            description="Plot effects from parcel-wise models.")
    
    parser.add_argument("results_df", 
                        help="File path to data frame storing parcel-wise model fits.")
    
    parser.add_argument("--effect", default='interaction_session2_parcel', 
                        help="Effect of interest")

    parser.add_argument("--outcomes", nargs='+', default=["qids_tot", "pcl5_tot", 
                                                          "vr36_emo_wellbeing", 
                                                          "vr36_energy", 
                                                          "vr36_role_limit_emo", 
                                                          "vr36_soc_fx"],
                        help="Outcome variables to test.")
    
    parser.add_argument("--vars", nargs='+', default=[], help="Variables used to determine direction of effect")
    parser.add_argument("--cluster", action="store_true", help="Whether to plot cluster-wise effects.")
    parser.add_argument("--alpha", default=0.05, type=float, help="Cluster-forming threshold used.")
    parser.add_argument("--cluster_df", 
                        help="Data frame storing clusters of significant " \
                        "parcels and associated p-values")
    parser.add_argument("--cluster_stat", default="psize")
    parser.add_argument("--F_stat", action='store_true')
    parser.add_argument("--corrected", action='store_true')
    parser.add_argument("--see_all", action='store_true')
    parser.add_argument("--fc_threshold", type=str, default="95thPercentile",
                        help="Connectivity threshold used to define parcel adjacency.")
    parser.add_argument("--svg", action='store_true', help="Save figures as svg files.")
    parser.add_argument("--t_max", type=float, help="Absolute value of boundary for color bar.")
    parser.add_argument("--cmap", default='RdBu_r')
    parser.add_argument("--plot_boundaries", action='store_true')

    # plot parcel wise FC for a single subject & session
    parser.add_argument("--plot_fc", action='store_true')
    parser.add_argument("--fc_df")
    parser.add_argument("--sub")
    parser.add_argument("--ses", type=int)

    # plot mask in schaefer atlas
    parser.add_argument("--plot_mask", action='store_true')
    parser.add_argument("--parcels", nargs='+', default=[], type=int)
    parser.add_argument("--mask_name")
    parser.add_argument("--mask_color")

    # plot model coefficients in schaefer atlas
    parser.add_argument("--plot_coef", action='store_true')
    parser.add_argument("--factors", nargs='+', type=float)
    parser.add_argument("--label")
    parser.add_argument("--mask_vars", nargs='+')
    parser.add_argument("--mask", action='store_true')

    return parser

if __name__ == "__main__":

    parser = generate_parser()
    args = parser.parse_args()

    results_df = pd.read_excel(args.results_df)

    if args.cluster_df is not None:
        cluster_df = pd.read_excel(args.cluster_df)
    
    flip_valence = {"qids_tot" : True, "pcl5_tot" : True, 
                    "vr36_energy" : False, "vr36_role_limit_emo" : False, 
                    "vr36_soc_fx" : False, "vr36_emo_wellbeing" : False, 
                    "parcel_fc" : False}
    
    if args.plot_fc: 
        fc_df = pd.read_excel(args.fc_df)

        plot_fc(fc_df, args.sub, args.ses, cmap=args.cmap)

    if args.plot_mask:
        plot_mask(args.parcels, args.mask_name, args.mask_color)

    for outcome in args.outcomes:
        print("OUTCOME: ", outcome, "\n", "-"*80)
        if args.plot_coef:
            plot_coefs(results_df, outcome, args.vars, args.label, 
                    factors=args.factors, mask=args.mask, 
                    mask_vars=args.mask_vars, c_max=args.t_max)
        elif args.cluster:
            plot_cluster_effects(cluster_df, results_df, outcome, args.effect, 
                             stat=args.cluster_stat, alpha=args.alpha,
                             vars=args.vars, 
                             flip_valence = flip_valence[outcome], 
                             see_all=args.see_all, svg=args.svg, 
                             t_max=args.t_max, cmap = args.cmap,
                             plot_boundaries=args.plot_boundaries)
        else:
            plot_effects(results_df, outcome, effect=args.effect, 
                        F_stat=args.F_stat, alpha=args.alpha, 
                        flip_valence = flip_valence[outcome], 
                        cmap=args.cmap, see_all=args.see_all,
                        corrected=args.corrected, vars=args.vars,
                        plot_boundaries=args.plot_boundaries, 
                        t_max=args.t_max)