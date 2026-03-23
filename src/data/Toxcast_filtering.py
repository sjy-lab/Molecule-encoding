"""
Created on Wed Jan  5 11:45:07 2022

@author: Moritz
"""

import pandas as pd


df_toxcast = pd.read_csv('/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding2/src/data/Processed_dataset/ToxCast_aggregated.csv')

#keep all assays with at least 50 actives and 50 inactives
assays_to_keep = []

for col in df_toxcast.columns[:-1]:
    counts = df_toxcast[col].value_counts()
    # robustly get counts for inactive (0) and active (1), covering 0/1, 0.0/1.0, '0'/'1', and booleans
    num_inactive = int(counts.get(0, 0)) + int(counts.get(0.0, 0)) + int(counts.get('0', 0)) + int(counts.get(False, 0))
    num_active = int(counts.get(1, 0)) + int(counts.get(1.0, 0)) + int(counts.get('1', 0)) + int(counts.get(True, 0))
    if num_inactive >= 50 and num_active >= 50:
        assays_to_keep.append(col)
        
assays_to_keep.append('standardised_smiles')

df_toxcast_filtered = df_toxcast.loc[:,assays_to_keep].copy()

df_toxcast_filtered.to_csv('/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding2/src/data/Processed_dataset/ToxCast_filtered.csv',index=False)