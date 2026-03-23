import argparse
import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from rdkit import RDLogger
from datetime import datetime

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')


def custom_tokenizer(text):
    return re.split(r'-|\\|/|\?|/~|=|#|~|@|:', text)


def main():
    parser = argparse.ArgumentParser(prog='SMARTS transform', description='select dataset and output directory')
    parser.add_argument('--dataset_name', type=str, default='Tox21', help="Select dataset")
    parser.add_argument('--dataset_path', type=str, default='/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/src/data/Processed_dataset/Tox21_aggregated.csv', help="Select dataset path")
    #parser.add_argument('--output_dim', type=int, default=128, help="Output dimension")
    parser.add_argument('--output_dir', type=str, default='/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/data_preprocess/InChI', help="Select output directory")

    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    DATASET_PATH = args.dataset_path
    SMARTS_OUTPUT_DIR = args.output_dir

    if not os.path.exists(DATASET_PATH):
        print(f"Error: file not found {DATASET_PATH}")
        exit(1)

    df = pd.read_csv(DATASET_PATH)
    smiles_list = df['standardised_smiles'].tolist()
    mol_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Filter invalid molecules and record indices
    original_count = len(mol_list)
    mol_list = [mol for mol in mol_list if mol is not None]
    num_invalid = original_count - len(mol_list)
    print(f"Successfully loaded {DATASET_NAME} dataset: {len(mol_list)} molecules")

    # Generate SMARTS strings
    smarts_features = []
    for mol in mol_list:
        try:
            smarts_str = Chem.MolToSmarts(mol)
            smarts_features.append(smarts_str)
        except Exception as e:
            print(f"Error generating SMARTS: {str(e)}")
            smarts_features.append("")

    print(f"Successfully generated {len(smarts_features)} SMARTS strings")

    # Use CountVectorizer for tokenization and vectorization
    cv = CountVectorizer(
        tokenizer=custom_tokenizer,
        lowercase=False,
        binary=False
    )

    X = cv.fit_transform(smarts_features)
    print(f'SMARTS vocabulary size: {len(cv.get_feature_names_out())}')
    print(f'SMARTS original feature shape: {X.shape}')

    # Use TF-IDF transformation
    transformer = TfidfTransformer()
    smarts_tfidf = transformer.fit_transform(X).toarray()
    print(f'SMARTS TF-IDF feature shape: {smarts_tfidf.shape}')

    # Create output directory and save as .npz format (fingerprints, smiles, version, timestamp, dataset_name, fingerprint_type, num_samples, num_invalid, vocab_size)
    os.makedirs(SMARTS_OUTPUT_DIR, exist_ok=True)
    smarts_file = os.path.join(SMARTS_OUTPUT_DIR, f"{DATASET_NAME}_SMARTS_embeddings.npz")
    
    np.savez_compressed(
        smarts_file,
        fingerprints=smarts_tfidf,
        smiles=np.array(smiles_list, dtype=object),
        version='1.0',
        timestamp=datetime.now().isoformat(),
        dataset_name=DATASET_NAME,
        fingerprint_type='SMARTS',
        num_samples=len(smiles_list),
        num_invalid=num_invalid,
        vocab_size=len(cv.get_feature_names_out())
    )

    print(f" SMARTS fingerprints saved to: {smarts_file}")
    print(f" Fingerprint shape: {smarts_tfidf.shape}")
    print(f" Format: .npz with SMILES validation support")

if __name__ == "__main__":
    main()