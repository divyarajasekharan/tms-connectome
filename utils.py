import pandas as pd

SITE_MAP = {'1991' : 'PA', 
            '1976' : 'PV', 
            '2000' : 'MN', 
            '1984' : 'WJ', 
            '1992' : 'MY'
}

def redcap_to_bids(redcap_id):
    
    site_id, sub_id = redcap_id.split("-")

    return "TMS" + SITE_MAP[site_id] + sub_id.zfill(3)

def combine_atlases(atlases, merge_on):

    df = pd.read_excel(atlases[0])

    for atlas in atlases[1:]: 
        atlas = pd.read_excel(atlas)
        df = pd.merge(df, atlas, on = merge_on)
        
    return df

def load_df(df_fp):
    if "xlsx" in df_fp:
        return pd.read_excel(df_fp)
    elif "csv" in df_fp:
        return pd.read_csv(df_fp)
    else: 
        return None 