from rdkit import RDLogger
from src.configs.dataset_configs import FINGERPRINT_CONFIGS
from src.datasets.base_dataset import BaseMolecularDataset
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


RDLogger.DisableLog('rdApp.warning')


def create_temp_dataset_for_fingerprints(smiles_list, fingerprint_type, dataset_name=None):
    class TempDataset(BaseMolecularDataset):
        def __init__(self, fingerprint_type, dataset_name=None):
            super().__init__(fingerprint_type=fingerprint_type)
            self.dataset_name = dataset_name
            if fingerprint_type in ['PubChem', 'SMARTS']:
                self.fingerprint_config = FINGERPRINT_CONFIGS[fingerprint_type]

    temp_dataset = TempDataset(fingerprint_type, dataset_name)

    from rdkit import Chem

    mols = []
    valid_indices = []
    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)

    if len(mols) == 0:
        raise ValueError("No valid SMILES found in the input list")

    features = temp_dataset.extract_features(mols)

    if len(mols) != len(smiles_list):
        feature_dim = len(features[0]) if len(features) > 0 else 1024
        full_features = np.zeros(
            (len(smiles_list), feature_dim), dtype=np.float32)
        for i, valid_idx in enumerate(valid_indices):
            full_features[valid_idx] = features[i]
        return full_features

    return np.array(features, dtype=np.float32)


def prepare_feature_network_data(df_filled, target_assay, fingerprint_type, smiles_column='standardised_smiles', dataset_name=None):
    """
    Prepare feature network data (fingerprint + other assays)

    Args:
        df_filled: DataFrame with imputed data
        target_assay: Target assay
        fingerprint_type: Fingerprint type
        smiles_column: SMILES column name
        dataset_name: Dataset name

    Returns:
        tuple: (X_features, y_target, available_indices)
    """

    available_indices = []
    for i, val in enumerate(df_filled[target_assay]):
        if not pd.isna(val):
            available_indices.append(i)

    smiles_list = df_filled[smiles_column].tolist()
    if fingerprint_type in ['PubChem', 'SMARTS']:
        X_fingerprints = create_temp_dataset_for_fingerprints(
            smiles_list, fingerprint_type, dataset_name=dataset_name)
    else:
        X_fingerprints = create_temp_dataset_for_fingerprints(
            smiles_list, fingerprint_type)

    other_assays = [col for col in df_filled.columns if col !=
                    target_assay and col != smiles_column]
    binary_assays = []
    for col in other_assays:
        col_series = df_filled[col]
        # try to convert to numeric, non-numeric will be converted to NaN and excluded
        col_numeric = pd.to_numeric(col_series, errors='coerce')
        non_nan = col_numeric.dropna()
        if non_nan.empty:
            continue
        unique_vals = pd.unique(non_nan)
        # only accept values in {0,1}
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            binary_assays.append(col)

    if len(binary_assays) > 0:
        X_other_assays = df_filled[binary_assays].values
    else:
        X_other_assays = np.empty((len(df_filled), 0), dtype=np.float32)

    print(f"X_fingerprints shape: {X_fingerprints.shape}")
    print(f"X_other_assays shape: {X_other_assays.shape}")

    X_features = np.concatenate([X_fingerprints, X_other_assays], axis=1)
    print(f"X_features shape: {X_features.shape}")

    y_target = df_filled[target_assay].values.astype(np.float32)

    X_features = X_features[available_indices]
    y_target = y_target[available_indices]

    X_features = np.array(X_features, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)

    return X_features, y_target, available_indices


def prepare_fingerprint_only_data(df, target_assay, fingerprint_type, smiles_column='standardised_smiles', dataset_name=None):
    """
    Only use molecular fingerprints as input features, without concatenating other assays.
    Returns: (X_fingerprints, y_target, available_indices)
    """
    available_indices = [i for i, val in enumerate(
        df[target_assay]) if not pd.isna(val)]

    smiles_list = df[smiles_column].tolist()
    if fingerprint_type in ['PubChem', 'SMARTS']:
        X_fingerprints = create_temp_dataset_for_fingerprints(
            smiles_list, fingerprint_type, dataset_name=dataset_name)
    else:
        X_fingerprints = create_temp_dataset_for_fingerprints(
            smiles_list, fingerprint_type)

    y_target = df[target_assay].values.astype(np.float32)

    X_fingerprints = np.array(X_fingerprints, dtype=np.float32)[
        available_indices]
    y_target = np.array(y_target, dtype=np.float32)[available_indices]

    return X_fingerprints, y_target, available_indices


def detect_task_type(y: np.ndarray) -> str:
    unique_values = np.unique(y)
    if len(unique_values) <= 10 and set(unique_values.tolist()).issubset({0, 1, 0.0, 1.0}):
        return 'classification'
    return 'regression'


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob > 0.5).astype(int)
    metrics = {}

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['auc'] = float(roc_auc_score(y_true, y_prob)) if len(
        np.unique(y_true)) == 2 else 0.0
    # zero_division=0 avoid warning
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['precision'] = float(
        precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}

    mse = mean_squared_error(y_true, y_pred)
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['mse'] = float(mse)
    metrics['r2'] = float(r2_score(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(mse))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Random Forest (sklearn API) on molecular datasets (fingerprint-only) with 5-fold 70/10/20 split',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., BBBP, MUTAG, ClinTox, Tox21, ToxCast, SIDER, BACE, ESOL)')
    parser.add_argument('--fingerprint', type=str, required=True,
                        choices=list(FINGERPRINT_CONFIGS.keys()), help='Fingerprint type')
    parser.add_argument('--multi_assay_data', type=str, required=True,
                        help='Aggregated CSV path (with standardised_smiles and assays)')
    parser.add_argument('--target_assay', type=str,
                        required=True, help='Target assay to predict')
    parser.add_argument('--smiles_column', type=str,
                        default='standardised_smiles', help='SMILES column name')

    parser.add_argument('--results_path', type=str,
                        default='experiments/results_RF_sklearn', help='Results output directory')
    parser.add_argument('--n_splits', type=int,
                        default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--requires_imputation',
                        action='store_true', help='Whether to use imputed data')
    parser.add_argument('--use_feature_net', action='store_true',
                        help='Whether to use feature network (fingerprint + other assays)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_path, exist_ok=True)

    df_raw = pd.read_csv(args.multi_assay_data)
    if args.target_assay not in df_raw.columns:
        raise ValueError(
            f"Target assay '{args.target_assay}' does not exist in the provided CSV")

    if args.requires_imputation:
        print("Need to impute")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(current_dir)))
        save_path = os.path.join(
            imputed_data_dir, f'{args.dataset}_{args.fingerprint}_imputed.csv')
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"Imputed data file not found: {save_path}. Please run the imputation process first, then try again."
            )
        df_imputed = pd.read_csv(save_path)

        if args.use_feature_net:
            print("Using feature network")
            X, y, available_indices = prepare_feature_network_data(
                df_imputed, args.target_assay, args.fingerprint, smiles_column=args.smiles_column, dataset_name=args.dataset
            )
        else:
            print("Not using feature network")
            X, y, available_indices = prepare_fingerprint_only_data(
                df_imputed, args.target_assay, args.fingerprint, smiles_column=args.smiles_column, dataset_name=args.dataset
            )
    else:
        print("No need to impute")
        # no need to impute, directly use original data
        if args.use_feature_net:
            print("Using feature network")
            X, y, available_indices = prepare_feature_network_data(
                df_raw, args.target_assay, args.fingerprint, smiles_column=args.smiles_column, dataset_name=args.dataset
            )
        else:
            print("Not using feature network")
            X, y, available_indices = prepare_fingerprint_only_data(
                df_raw, args.target_assay, args.fingerprint, smiles_column=args.smiles_column, dataset_name=args.dataset
            )

    # check if there are available data
    if len(available_indices) == 0:
        raise ValueError('All target labels are missing, cannot evaluate')

    X = np.array(X, dtype=np.float32)
    if X.shape[0] != len(y):
        raise ValueError(
            'Feature and label counts do not match, please check the fingerprint generation process')

    task_type = detect_task_type(y)
    print(f"Detected task type: {task_type}")

    rows = []
    if task_type == 'classification':
        kfold = StratifiedKFold(n_splits=args.n_splits,
                                shuffle=True, random_state=args.seed)
        splits = kfold.split(X, y)
    else:
        kfold = KFold(n_splits=args.n_splits,
                      shuffle=True, random_state=args.seed)
        splits = kfold.split(X)

    for fold_id, (trainval_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nExecuting fold {fold_id}:")
        X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if task_type == 'classification':
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=1/8, random_state=args.seed, stratify=y_trainval
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=1/8, random_state=args.seed
            )
            # regression task: standardize target variable
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(
                y_train.reshape(-1, 1)).astype(np.float32)
            y_val_scaled = scaler.transform(
                y_val.reshape(-1, 1)).astype(np.float32)

        if task_type == 'classification':
            # scale_pos_weight to handle imbalance
            pos_count = float((y_train == 1).sum())
            neg_count = float((y_train == 0).sum())
            scale_pos_weight = (
                neg_count / pos_count) if pos_count > 0 else 1.0
            model = RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,
                class_weight='balanced',
                random_state=args.seed
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                criterion='friedman_mse',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,
                random_state=args.seed
            )

        if task_type == 'classification':
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train_scaled.ravel())

        if task_type == 'classification':
            val_prob = model.predict_proba(X_val)[:, 1]
            val_metrics = compute_classification_metrics(y_val, val_prob)
        else:
            val_pred_scaled = model.predict(X_val)
            val_pred = scaler.inverse_transform(
                val_pred_scaled.reshape(-1, 1)).ravel()
            val_metrics = compute_regression_metrics(y_val, val_pred)

        if task_type == 'classification':
            test_prob = model.predict_proba(X_test)[:, 1]
            test_metrics = compute_classification_metrics(y_test, test_prob)
        else:
            test_pred_scaled = model.predict(X_test)
            test_pred = scaler.inverse_transform(
                test_pred_scaled.reshape(-1, 1)).ravel()
            test_metrics = compute_regression_metrics(y_test, test_pred)

        val_row = {
            'dataset': args.dataset,
            'fingerprint': args.fingerprint,
            'target_assay': args.target_assay,
            'fold': fold_id,
            'set': 'val',
            **val_metrics
        }
        test_row = {
            'dataset': args.dataset,
            'fingerprint': args.fingerprint,
            'target_assay': args.target_assay,
            'fold': fold_id,
            'set': 'test',
            **test_metrics
        }
        rows.extend([val_row, test_row])

        per_fold_out = os.path.join(
            args.results_path,
            f"{args.dataset}_{args.fingerprint}_{args.target_assay}_RF_fold{fold_id}_preds.csv"
        )
        if task_type == 'classification':
            pred_df = pd.DataFrame({
                'set': ['val'] * len(y_val) + ['test'] * len(y_test),
                'true': np.concatenate([y_val, y_test]).astype(float),
                'pred_proba': np.concatenate([val_prob, test_prob]).astype(float)
            })
            pred_df[['true', 'pred_proba']] = pred_df[[
                'true', 'pred_proba']].round(3)
            pred_df.to_csv(per_fold_out, index=False)
        else:
            pred_df = pd.DataFrame({
                'set': ['val'] * len(y_val) + ['test'] * len(y_test),
                'true': np.concatenate([y_val, y_test]).astype(float),
                'pred': np.concatenate([val_pred, test_pred]).astype(float)
            })
            pred_df[['true', 'pred']] = pred_df[['true', 'pred']].round(3)
            pred_df.to_csv(per_fold_out, index=False)

    metrics_df = pd.DataFrame(rows)

    base_cols = ['dataset', 'fingerprint', 'target_assay', 'fold', 'set']
    if task_type == 'classification':
        metric_cols = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    else:
        metric_cols = ['mae', 'mse', 'r2', 'rmse']

    metrics_df = metrics_df[base_cols + metric_cols]

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_df[numeric_cols] = metrics_df[numeric_cols].round(3)
    metrics_out = os.path.join(
        args.results_path,
        f"{args.dataset}_{args.fingerprint}_{args.target_assay}_RF_folds_metrics.csv"
    )
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved per fold evaluation metrics: {metrics_out}")

    def agg_metrics(df_sub: pd.DataFrame, task_type: str) -> pd.DataFrame:
        # exclude fold and set, only aggregate on metric columns
        metric_cols = [col for col in df_sub.select_dtypes(
            include=[np.number]).columns if col != 'fold']

        # group by set and calculate mean and std
        grouped = df_sub.groupby(['set'])[metric_cols].agg(['mean', 'std'])

        result_data = {'set': grouped.index.tolist()}

        if task_type == 'classification':
            ordered_metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
        else:
            ordered_metrics = ['mae', 'mse', 'r2', 'rmse']

        for col in ordered_metrics:
            if col in metric_cols:
                mean_vals = grouped[(col, 'mean')].values
                std_vals = grouped[(col, 'std')].values
                result_data[col] = [
                    f"{mean:.3f} ± {std:.3f}" for mean, std in zip(mean_vals, std_vals)]

        return pd.DataFrame(result_data)

    summary = agg_metrics(metrics_df, task_type)
    summary_out = os.path.join(
        args.results_path,
        f"{args.dataset}_{args.fingerprint}_{args.target_assay}_RF_summary.csv"
    )
    summary.to_csv(summary_out, index=False)
    print(f"Saved summary results: {summary_out}")


if __name__ == '__main__':
    main()
