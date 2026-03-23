import pandas as pd
from pathlib import Path

_BASE = Path(__file__).resolve().parents[4]

smiles_path = _BASE / 'src/data/Original_dataset/mutag_188_data.can'
labels_path = _BASE / 'src/data/Original_dataset/mutag_188_target.txt'
output_csv_path = _BASE / 'src/data/Original_dataset/mutag_combined.csv'

smiles = pd.read_csv(smiles_path, sep=' ', header=None)[0]
labels = pd.read_csv(labels_path, header=None)[0]

labels[labels == -1] = 0

df = pd.DataFrame({'smiles': smiles, 'label': labels})

df.to_csv(output_csv_path, index=False)

