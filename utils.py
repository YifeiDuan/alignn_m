import pandas as pd
import numpy as np
import math
import random
import os

def concat_id_prop_csv(prop_name="matbench_jdft2d_exfoliation_en"):
    df_all = pd.DataFrame()
    
    for fold in range(5):
        df_fold = pd.read_csv(f"alignn/{prop_name}_fold_{fold}/id_prop.csv", header=None).rename(columns={0:"id", 1:"target"})
        
        df_all = pd.concat([df_all, df_fold], ignore_index=True)
    
    df_all = df_all.drop_duplicates(subset=["id"]).sort_values(by=["id"])
    
    save_dir = f"text/{prop_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    df_all.to_csv(f"{save_dir}/id_prop_all.csv", index=False)