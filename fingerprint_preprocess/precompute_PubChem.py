import argparse
import os
import pandas as pd
import numpy as np
import pubchempy as pcp
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(prog='PubChem transform', description='select dataset and output directory')
    parser.add_argument('--dataset_name', type=str, default='tox21', help="Select dataset")
    parser.add_argument('--dataset_path', type=str, default='/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/src/data/Processed_dataset/Tox21_aggregated.csv', help="Select dataset path")
    parser.add_argument('--output_dir', type=str, default='/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/data_preprocess/PubChem', help="Select output directory")

    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    DATASET_PATH = args.dataset_path
    PUBCHEM_OUTPUT_DIR = args.output_dir

    os.makedirs(PUBCHEM_OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(DATASET_PATH)  # Read the current file
    smiles_column = "standardised_smiles"

    # Extract SMILES from the specified column
    smiles_list = df[smiles_column].tolist()

    fingerprints = []
    missing_fingerprints = 0
    error_fingerprints = 0
    success_fingerprints = 0

    for smile in smiles_list:
        try:
            pubchem_compound = pcp.get_compounds(smile, 'smiles')[0]

            if pubchem_compound.fingerprint is None:
                print(f"Missing fingerprint for compound: {pubchem_compound}")

                fingerprints.append([0] * 881)  # Default to zero fingerprint or handle differently
                missing_fingerprints += 1
            
            else:

                # Extract the fingerprint if available as a list of bits
                feature = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]

                # Append the fingerprint to the list
                fingerprints.append(feature)
                print(f'fingerprint : {smile}\n')

                success_fingerprints += 1


        except Exception as e:
            fingerprints.append([0] * 881)
            error_fingerprints += 1

    fingerprints = np.asarray(fingerprints)

    print(f"\nResults Summary:")
    print(f"Fingerprints shape: {fingerprints.shape}")
    print(f"Missing fingerprints: {missing_fingerprints}")
    print(f"Error fingerprints: {error_fingerprints}")
    print(f"Success fingerprints: {success_fingerprints}")
    print(f"Total processed: {len(smiles_list)}")

    # Save embeddings as .npz format (fingerprints, smiles, version, timestamp, dataset_name, fingerprint_type, num_samples, num_missing, num_errors, num_success)
    output_file = f"{PUBCHEM_OUTPUT_DIR}/{DATASET_NAME}_PubChem_embeddings.npz"
    np.savez_compressed(
        output_file,
        fingerprints=fingerprints,
        smiles=np.array(smiles_list, dtype=object),
        version='1.0',
        timestamp=datetime.now().isoformat(),
        dataset_name=DATASET_NAME,
        fingerprint_type='PubChem',
        num_samples=len(smiles_list),
        num_missing=missing_fingerprints,
        num_errors=error_fingerprints,
        num_success=success_fingerprints
    )
    print(f"\n Embeddings saved to {output_file}")
    print(f"   Format: .npz with SMILES validation support")

if __name__ == "__main__":
    main()