from src.configs.dataset_configs import FINGERPRINT_CONFIGS
from src.datasets.base_dataset import BaseMolecularDataset
import os
import sys
import argparse
import numpy as np
import pandas as pd
import random
from xgboost import XGBClassifier, XGBRegressor
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def set_global_seed(seed: int = 42):
    """set global seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def create_temp_dataset_for_fingerprints(smiles_list, fingerprint_type, dataset_name=None):
    """
    precompute fingerprint(same as main flow)
    """

    class TempDataset(BaseMolecularDataset):
        def __init__(self, fingerprint_type, dataset_name=None):
            super().__init__(fingerprint_type=fingerprint_type)
            self.dataset_name = dataset_name
            if fingerprint_type in ['PubChem', 'InChI', 'SMARTS']:
                self.fingerprint_config = FINGERPRINT_CONFIGS[fingerprint_type]

    temp_dataset = TempDataset(fingerprint_type, dataset_name)

    # transform smiles, make sure all smiles are valid
    mols = []
    valid_indices = []
    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
        else:
            print(f"Warning: Invalid SMILES: {smile}")

    if len(mols) == 0:
        raise ValueError("No valid SMILES strings")

    # extract features
    features = temp_dataset.extract_features(mols)

    # if there are invalid molecules, fill with zero vectors
    if len(mols) != len(smiles_list):
        print(
            f"Warning: {len(smiles_list) - len(mols)} molecules cannot be parsed, filling with zero vectors")
        feature_dim = len(features[0]) if len(features) > 0 else 1024
        full_features = np.zeros((len(smiles_list), feature_dim))
        for i, valid_idx in enumerate(valid_indices):
            full_features[valid_idx] = features[i]
        return full_features

    return np.array(features)


def impute_missing_values(df_data, fingerprint_type, smiles_column='standardised_smiles', save_path=None, args=None):
    """
    Impute missing values using XGBoost (can handle binary classification and regression), standalone callable.

    Args:
        df_data: DataFrame containing SMILES and assay data
        fingerprint_type: fingerprint type
        smiles_column: SMILES column name
        save_path: path to save the imputed data
        args: other parameters (when using precomputed fingerprints, provide args.dataset as dataset_name)
    """

    df_filled = df_data.copy()

    # assays to impute (exclude smiles column)
    assays = [col for col in df_data.columns if col != smiles_column]

    # generate fingerprints for all molecules
    smiles_list = df_data[smiles_column].tolist()
    if fingerprint_type in ['InChI', 'PubChem', 'SMARTS']:
        actual_dataset_name = getattr(args, 'dataset', 'MultiAssayDataset')
        X_all = create_temp_dataset_for_fingerprints(
            smiles_list, fingerprint_type, dataset_name=actual_dataset_name)
    else:
        X_all = create_temp_dataset_for_fingerprints(
            smiles_list, fingerprint_type)

    X_all = np.array(X_all, dtype=np.float32)
    print(
        f"Fingerprint dimension: {X_all.shape}, assays to impute: {len(assays)}")

    for assay in assays:
        print(f"\nProcessing assay: {assay}")
        valid_indices = df_data[assay].notna()

        if valid_indices.sum() == 0:
            print(
                f"Warning: assay {assay} has no available training data, skipping imputation")
            continue

        # check if binary or continuous
        unique_values = df_data[assay][valid_indices].unique()
        is_binary = all(val in [0, 1] for val in unique_values)
        if not is_binary and not np.issubdtype(df_data[assay].dtype, np.number):
            print(
                f"Warning: assay {assay} contains non-numeric/non-binary values {unique_values}, skipping imputation")
            continue

        X_train = X_all[valid_indices]
        y_train = df_data[assay][valid_indices].values.astype(np.float32)

        print(f"Training data size: X={X_train.shape}, y={y_train.shape}")
        unique_values = np.unique(y_train)
        print(f"  {assay} unique values: {unique_values}")
        print(f"  {assay} value range: [{y_train.min()}, {y_train.max()}]")

        # determine task type
        if len(unique_values) == 2 and all(val in [0, 1] for val in unique_values):
            task_type = 'classification'
        else:
            task_type = 'regression'

        try:
            if task_type == 'classification':
                pos_count = float((y_train == 1).sum())
                neg_count = float((y_train == 0).sum())
                scale_pos_weight = (
                    neg_count / pos_count) if pos_count > 0 else 1.0
                model = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    reg_alpha=1,
                    subsample=1.0,
                    tree_method='hist',
                    random_state=getattr(args, 'seed', 42),
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            else:
                model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    reg_alpha=1,
                    subsample=1.0,
                    tree_method='hist',
                    random_state=getattr(args, 'seed', 42),
                    n_jobs=-1,
                    eval_metric='rmse'
                )

            model.fit(X_train, y_train)

            predictions = model.predict(X_all)

            missing_idx = df_filled[assay].isna()

            df_filled.loc[missing_idx, assay] = predictions[missing_idx]

            print(f"Completed imputation for {assay}")

        except Exception as e:
            print(f"Error processing {assay}: {str(e)}")
            continue

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_filled.to_csv(save_path, index=False)
        print(f"\nImputed data saved to: {save_path}")

    return df_filled


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Impute missing values for a dataset (handling both target and other assays)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=str,
                        help="Dataset name (used for precomputed fingerprints and output naming)")
    parser.add_argument("--fingerprint", required=True, type=str,
                        choices=list(FINGERPRINT_CONFIGS.keys()), help="Fingerprint type")
    parser.add_argument("--input_csv", required=True,
                        type=str, help="Original multi-assay CSV path")
    parser.add_argument("--smiles_column", default="standardised_smiles",
                        type=str, help="SMILES column name")
    parser.add_argument("--results_path", default="experiments/results",
                        type=str, help="Result output root directory")
    parser.add_argument("--output", default=None, type=str,
                        help="Custom output CSV path (if not provided, use default naming)")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed (used for XGBoost and data splitting)")
    return parser.parse_args()


def _main_():
    args = _parse_args()

    # set random seed
    set_global_seed(args.seed)

    # read input data
    df = pd.read_csv(args.input_csv)

    # prepare output path
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(args.results_path, "imputed_data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{args.dataset}_{args.fingerprint}_imputed.csv")

    impute_missing_values(
        df_data=df,
        fingerprint_type=args.fingerprint,
        smiles_column=args.smiles_column,
        save_path=out_path,
        args=args,
    )

    print(f"Completed imputation, output: {out_path}")


if __name__ == "__main__":
    _main_()
