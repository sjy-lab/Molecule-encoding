import pandas as pd

# Upload CSV
file_path = "/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/src/data/Processed_dataset/BACE_aggregated.csv"
df = pd.read_csv(file_path)

missing_positions = df.isna()

# count number of missing
missing_counts = df.isna().sum()

columns_with_missing = missing_counts[missing_counts > 0]

print(columns_with_missing)
