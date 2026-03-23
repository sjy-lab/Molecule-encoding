#!/usr/bin/env python3
"""
Validate the completeness and correctness of .npz fingerprint files
Check if SMILES matches the CSV and ensure the order is aligned
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import sys

# Supported datasets and fingerprint types
DATASETS = ['BBBP', 'BACE', 'ClinTox', 'ESOL', 'MUTAG', 'SIDER', 'ToxCast']
FINGERPRINT_TYPES = ['PubChem', 'SMARTS', 'InChI']

def validate_npz_file(npz_path: Path, csv_path: Path, smiles_column: str = 'standardised_smiles') -> dict:
    """
    Validate a single .npz file
    
    Returns:
        dict: Validation result
    """
    result = {
        'file': str(npz_path),
        'exists': npz_path.exists(),
        'valid': False,
        'error': None,
        'metadata': {}
    }
    
    if not npz_path.exists():
        result['error'] = f"File not found: {npz_path}"
        return result
    
    if not csv_path.exists():
        result['error'] = f"CSV not found: {csv_path}"
        return result
    
    try:
        # Load .npz file
        data = np.load(npz_path, allow_pickle=True)
        
        # Check required keys
        required_keys = ['fingerprints', 'smiles']
        missing_keys = [k for k in required_keys if k not in data.files]
        if missing_keys:
            result['error'] = f"Missing required keys: {missing_keys}"
            return result
        
        # Extract data
        fingerprints = data['fingerprints']
        stored_smiles = data['smiles']
        print(f"fingerprints: {fingerprints.shape}")
        print(f"stored_smiles: {len(stored_smiles)}")   
        
        # Extract metadata
        metadata_keys = ['version', 'timestamp', 'dataset_name', 'fingerprint_type', 'num_samples']
        for key in metadata_keys:
            if key in data.files:
                result['metadata'][key] = str(data[key])
        
        # Load CSV
        df = pd.read_csv(csv_path)
        if smiles_column not in df.columns:
            result['error'] = f"Column not found in CSV: {smiles_column}"
            return result
        
        csv_smiles = df[smiles_column].tolist()
        print(f"csv_smiles: {len(csv_smiles)}")
        
        # Validate 1: row count一致
        if len(stored_smiles) != len(csv_smiles):
            result['error'] = f"Row count mismatch: .npz={len(stored_smiles)}, CSV={len(csv_smiles)}"
            return result
        
        # Validate 2: SMILES fully match
        mismatches = []
        for i, (stored, csv_s) in enumerate(zip(stored_smiles, csv_smiles)):
            if str(stored) != str(csv_s):
                mismatches.append(i)
                if len(mismatches) <= 5:  # Only record the first 5 mismatches
                    print(f"  Index {i}: .npz='{stored}' vs CSV='{csv_s}'")
        
        if mismatches:
            result['error'] = f"SMILES mismatch: {len(mismatches)} positions"
            result['mismatches'] = mismatches[:10]  # Only record the first 10 mismatches
            return result
        
        # Validate 3: fingerprint shape
        if fingerprints.shape[0] != len(csv_smiles):
            result['error'] = f"Fingerprint row count mismatch: {fingerprints.shape[0]} vs {len(csv_smiles)}"
            return result
        
        # All passed
        result['valid'] = True
        result['fingerprint_shape'] = fingerprints.shape
        result['num_samples'] = len(csv_smiles)
        
    except Exception as e:
        result['error'] = f"Validation error: {str(e)}"
    
    return result


def validate_all_datasets(
    datasets: List[str],
    fingerprint_types: List[str],
    csv_dir: Path,
    fp_dir: Path,
    smiles_column: str = 'standardised_smiles'
) -> dict:
    """
    Validate all datasets and fingerprint types
    
    Returns:
        dict: Overall validation result
    """
    results = {}
    total = len(datasets) * len(fingerprint_types)
    current = 0
    
    print(f"\n{'='*80}")
    print(f"Start validating {len(datasets)} datasets × {len(fingerprint_types)} fingerprint types = {total} files")
    print(f"{'='*80}\n")
    
    for dataset in datasets:
        results[dataset] = {}
        csv_path = csv_dir / f"{dataset}_aggregated.csv"
        if dataset == 'ToxCast':
            csv_path = csv_dir / f"{dataset}_filtered.csv"
        
        for fp_type in fingerprint_types:
            current += 1
            npz_path = fp_dir / fp_type / f"{dataset}_{fp_type}_embeddings.npz"
            
            print(f"[{current}/{total}] Validating {dataset} - {fp_type}...")
            
            result = validate_npz_file(npz_path, csv_path, smiles_column)
            results[dataset][fp_type] = result
            
            if result['valid']:
                print(f"     Passed validation")
                print(f"     Fingerprint shape: {result['fingerprint_shape']}")
                print(f"     Number of samples: {result['num_samples']}")
                if 'version' in result['metadata']:
                    print(f"     Version: {result['metadata']['version']}")
            else:
                print(f"  Failed validation: {result['error']}")
            print()
    
    return results


def print_summary(results: dict):
    """Print validation summary"""
    print(f"\n{'='*80}")
    print("Validation summary")
    print(f"{'='*80}\n")
    
    total_files = 0
    valid_files = 0
    missing_files = 0
    invalid_files = 0
    
    for dataset, fp_results in results.items():
        print(f"\n{dataset}:")
        for fp_type, result in fp_results.items():
            total_files += 1
            status = ""
            if not result['exists']:
                status = "檔案不存在"
                missing_files += 1
            elif result['valid']:
                status = f"通過 ({result['fingerprint_shape']})"
                valid_files += 1
            else:
                status = f"失敗: {result['error']}"
                invalid_files += 1
            
            print(f"  {fp_type:10s}: {status}")
    
    print(f"\n{'='*80}")
    print(f"Total: {total_files} files")
    print(f"  Passed: {valid_files}")
    print(f"  Failed: {invalid_files}")
    print(f"  Missing: {missing_files}")
    print(f"{'='*80}\n")
    
    if valid_files == total_files:
        print("All files passed validation!")
        return 0
    else:
        print("Some files failed validation, please check the detailed information above.")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Validate .npz fingerprint files')
    parser.add_argument('--csv_dir', type=Path,
                       default=Path('src/data/Processed_dataset'),
                       help='CSV file directory')
    parser.add_argument('--fp_dir', type=Path,
                       default=Path('fingerprint_preprocess'),
                       help='Fingerprint file directory')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                       choices=DATASETS,
                       help='Datasets to validate')
    parser.add_argument('--fingerprints', nargs='+', default=FINGERPRINT_TYPES,
                       choices=FINGERPRINT_TYPES,
                       help='Fingerprint types to validate')
    parser.add_argument('--smiles_column', type=str,
                       default='standardised_smiles',
                       help='SMILES column name')
    
    args = parser.parse_args()
    
    # Validate all files
    results = validate_all_datasets(
        datasets=args.datasets,
        fingerprint_types=args.fingerprints,
        csv_dir=args.csv_dir,
        fp_dir=args.fp_dir,
        smiles_column=args.smiles_column
    )
    
    # Print summary
    exit_code = print_summary(results)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

