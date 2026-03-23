from src.datasets.base_dataset import BaseMolecularDataset
from src.configs.dataset_configs import (
    get_model_config, DEFAULT_TRAINING_CONFIG,
    FINGERPRINT_TYPES
)
from src.utils.training_utils_improved import (
    set_seed, get_device, prep_dataloader,
    export_single_smiles_attention,
    train_kfold_classification_model, train_kfold_regression_model,
    train_single_fold_classification_model, train_single_fold_regression_model
)
from src.models.neural_networks import NeuralNet, TransformerMLPModel, init_weights
import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import math
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from imblearn.combine import SMOTEENN
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch.utils.data import DataLoader, TensorDataset
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def save_experiment_config(dataset, model_config, training_config, fingerprint_type=None, model_type=None, output_dir="experiments/configs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(
        output_dir, f"{dataset}_{timestamp}_config.json")

    config = {
        "dataset": dataset,
        "fingerprint_type": fingerprint_type,
        "model_type": model_type,
        "model_config": model_config.__dict__,
        "training_config": training_config.__dict__
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config_path, timestamp


def log_experiment_result(dataset, model_config, training_config, metrics, timestamp, output_file="experiments/results.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    row = {
        "timestamp": timestamp,
        "dataset": dataset,
        **{f"model_{k}": v for k, v in model_config.__dict__.items()},
        **{f"train_{k}": v for k, v in training_config.__dict__.items()},
        **metrics
    }

    file_exists = os.path.isfile(output_file)

    if file_exists:
        with open(output_file, 'r+b') as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                last_char = f.read(1)
                if last_char != b'\n':
                    f.write(b'\n')

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Training script for molecular embedding project',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset', type=str, default='MultiAssayDataset',
                        help='dataset name (for file naming)')
    parser.add_argument('--model_type', type=str, default='MLP',
                        choices=['MLP', 'MLP+TL'],
                        help='select model type')
    parser.add_argument('--fingerprint', type=str, default='MACCS',
                        choices=FINGERPRINT_TYPES,
                        help='select fingerprint type')
    parser.add_argument('--results_path', type=str, default='experiments',
                        help='result save path')
    parser.add_argument('--multi_assay_data', type=str, required=True,
                        help='path to multi-assay dataset')
    parser.add_argument('--target_assay', type=str, required=True,
                        help='target assay column name for feature network training')
    parser.add_argument('--smiles_column', type=str, default='standardised_smiles',
                        help='SMILES column name (default: standardised_smiles)')
    parser.add_argument('--use_smoteenn', action='store_true',
                        help='use SMOTEENN for classification tasks (when data is imbalanced)')
    parser.add_argument('--imbalance_threshold', type=float, default=0.3,
                        help='imbalance threshold, use SMOTEENN when minority class ratio is less than this value')
    parser.add_argument('--use_input_layernorm',
                        action='store_true', help='use input layernorm')
    parser.add_argument('--use_single_fold',
                        action='store_true', help='use single fold training')
    parser.add_argument('--id_col', type=str, default='standardised_smiles',
                        help='SMILES column name (default: standardised_smiles)')
    parser.add_argument('--test_id', type=str, default=None,
                        help='specify single SMILES (standardised_smiles) to output last layer head-avg attention heatmap')
    parser.add_argument('--verify_transformer', action='store_true',
                        help='verify transformer intermediate tensors and attention weights (only for MLP+TL and provided --test_id)')
    parser.add_argument('--use_warmup', action='store_true',
                        help='use learning rate warm-up (linear increase LR for first 50 epochs)')

    return parser.parse_args()


def apply_smoteenn(X, y):

    try:
        smoteenn = SMOTEENN(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)
        print(
            f"SMOTEENN resampling before: {np.unique(y, return_counts=True)}")
        print(
            f"SMOTEENN resampling after: {np.unique(y_resampled, return_counts=True)}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"SMOTEENN processing failed: {e}")
        print("continue using original data...")
        return X, y


def check_class_imbalance(y, threshold=0.3):
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2:
        return False  # not a binary classification task, no need to use SMOTEENN

    minority_ratio = min(counts) / sum(counts)
    print(
        f"class distribution: {dict(zip(unique, counts))}, minority class ratio: {minority_ratio:.3f}")

    return minority_ratio < threshold


def create_temp_dataset_for_fingerprints(smiles_list, fingerprint_type, dataset_name=None):
    """create temporary dataset object to generate fingerprints"""

    class TempDataset(BaseMolecularDataset):
        def __init__(self, fingerprint_type, dataset_name=None):
            super().__init__(fingerprint_type=fingerprint_type)
            self.dataset_name = dataset_name
            # set configuration for all fingerprint types (to avoid undefined attributes)
            from src.configs.dataset_configs import FINGERPRINT_CONFIGS
            self.fingerprint_config = FINGERPRINT_CONFIGS.get(fingerprint_type)

    temp_dataset = TempDataset(fingerprint_type, dataset_name)

    mols = []
    valid_indices = []

    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
        else:
            print(f"warning: cannot parse SMILES: {smile}")

    if len(mols) == 0:
        raise ValueError("no valid SMILES strings")

    features = temp_dataset.extract_features(mols)

    # if there are invalid molecules, create a full-size feature array and fill with zeros
    if len(mols) != len(smiles_list):
        print(
            f"warning: {len(smiles_list) - len(mols)} SMILES cannot be parsed, will use zero vector")

        feature_dim = len(features[0]) if len(features) > 0 else 1024
        full_features = np.zeros((len(smiles_list), feature_dim))

        for i, valid_idx in enumerate(valid_indices):
            full_features[valid_idx] = features[i]

        return full_features

    return np.array(features)


def prepare_feature_network_data(df_filled, target_assay, fingerprint_type, smiles_column='standardised_smiles', dataset_name=None):
    """
    prepare feature network data (fingerprints + other assay labels)

    Args:
        df_filled: imputed data frame
        target_assay: target assay
        fingerprint_type: fingerprint type
        smiles_column: SMILES column name
        dataset_name: dataset name (for InChI and PubChem fingerprints)

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
        col_numeric = pd.to_numeric(col_series, errors='coerce')
        non_nan = col_numeric.dropna()
        if non_nan.empty:
            continue
        unique_vals = pd.unique(non_nan)
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
    prepare fingerprint only data (fingerprints only)

    Args:
        df: data frame
        target_assay: target assay
        fingerprint_type: fingerprint type
        smiles_column: SMILES column name
        dataset_name: dataset name (for InChI and PubChem fingerprints)

    Returns:
        tuple: (X_fingerprints, y_target, available_indices)
    """
    available_indices = [i for i, val in enumerate(
        df[target_assay]) if not pd.isna(val)]

    smiles_list = df[smiles_column].tolist()
    if fingerprint_type in ['InChI', 'PubChem', 'SMARTS']:
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


def run_kfold_experiment(dataset_name, fingerprint_type, model_type, args, device):
    """run feature network experiment (imputed + fingerprints + other assay features)"""

    print(
        f"Running Feature Network experiment: {fingerprint_type} - {args.target_assay} - {model_type}\n")

    # record experiment configuration
    dataset_config, fingerprint_config = get_model_config(
        dataset_name, fingerprint_type, model_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save experiment configuration to JSON file
    config_path, _ = save_experiment_config(
        dataset=dataset_name,
        model_config=dataset_config.model_config,
        training_config=dataset_config.training_config,
        fingerprint_type=fingerprint_type,
        model_type=model_type,
        output_dir=os.path.join(
            args.results_path, dataset_name, model_type, "configs")
    )
    print(f"experiment configuration saved to: {config_path}")

    if not os.path.exists(args.multi_assay_data):
        print(
            f"error: multi-assay data file not found: {args.multi_assay_data}")
        return None

    df_raw = pd.read_csv(args.multi_assay_data)
    if 'Identifier' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Identifier'])

    if args.target_assay not in df_raw.columns:
        print(f"error: target assay '{args.target_assay}' not found in data")
        return None

    if getattr(dataset_config, 'requires_imputation', True):
        print("imputation required")
        base_dir = os.path.dirname(__file__)
        imputed_data_dir = os.path.join(
            base_dir, 'src', 'data', 'Imputed_data')
        save_path = os.path.join(
            imputed_data_dir, f'{dataset_name}_{fingerprint_type}_imputed.csv')
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"imputed data file not found: {save_path}. please run imputation process to generate this file, then try again."
            )
        df_imputed = pd.read_csv(save_path)

        if getattr(dataset_config, 'use_feature_net', True):
            print("using feature network")
            X_features, y_target, available_indices = prepare_feature_network_data(
                df_imputed, args.target_assay, fingerprint_type, smiles_column=args.smiles_column, dataset_name=dataset_name
            )
        else:
            print("not using feature network")
            X_features, y_target, available_indices = prepare_fingerprint_only_data(
                df_imputed, args.target_assay, fingerprint_type, smiles_column=args.smiles_column, dataset_name=dataset_name
            )
    else:
        print("no imputation required")
        # 不需要補值，直接使用原始資料
        if getattr(dataset_config, 'use_feature_net', True):
            print("using feature network")
            X_features, y_target, available_indices = prepare_feature_network_data(
                df_raw, args.target_assay, fingerprint_type, smiles_column=args.smiles_column, dataset_name=dataset_name
            )
        else:
            print("not using feature network")
            X_features, y_target, available_indices = prepare_fingerprint_only_data(
                df_raw, args.target_assay, fingerprint_type, smiles_column=args.smiles_column, dataset_name=dataset_name
            )

    task_type = dataset_config.task_type
    # initialize n_splits cross-validation
    n_splits = dataset_config.training_config.n_splits
    if task_type == 'classification':
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42)
        splits = kfold.split(X_features, y_target)
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = kfold.split(X_features)

    print(
        f"using {n_splits}-fold cross-validation, total data size: {len(X_features)} samples")

    for fold, (train_val_idx, test_idx) in enumerate(splits):
        # get current fold's train+val and test data
        X_train_val_fold = X_features[train_val_idx]
        y_train_val_fold = y_target[train_val_idx]
        X_test_fold = X_features[test_idx]
        y_test_fold = y_target[test_idx]

        # split train+val into train and val in 7:1 ratio (val占train+val的1/8 = 0.125)
        if task_type == 'classification':
            train_idx_inner, val_idx_inner = train_test_split(
                np.arange(len(X_train_val_fold)),
                test_size=0.125,  # 1/(7+1) = 0.125
                stratify=y_train_val_fold,
                random_state=42
            )
        else:
            train_idx_inner, val_idx_inner = train_test_split(
                np.arange(len(X_train_val_fold)),
                test_size=0.125,
                random_state=42
            )

        X_train_fold = X_train_val_fold[train_idx_inner]
        y_train_fold = y_train_val_fold[train_idx_inner]
        X_val_fold = X_train_val_fold[val_idx_inner]
        y_val_fold = y_train_val_fold[val_idx_inner]

        print(f"{fold + 1} fold data distribution:")
        print(f"  training set: {len(X_train_fold)} samples")
        print(f"  validation set: {len(X_val_fold)} samples")
        print(f"  test set: {len(X_test_fold)} samples")
        total_samples = len(X_train_fold) + len(X_val_fold) + len(X_test_fold)
        print(
            f"  ratio = {len(X_train_fold)/total_samples:.1f}:{len(X_val_fold)/total_samples:.1f}:{len(X_test_fold)/total_samples:.1f}")

        label_inverse_transform = None
        if task_type == 'regression' and dataset_name in ['ESOL']:
            scaler = StandardScaler()
            y_train_fold_scaled = scaler.fit_transform(
                y_train_fold.reshape(-1, 1)).astype(np.float32)
            y_val_fold_scaled = scaler.transform(
                y_val_fold.reshape(-1, 1)).astype(np.float32)
            label_inverse_transform = scaler.inverse_transform
        else:
            y_train_fold_scaled = y_train_fold
            y_val_fold_scaled = y_val_fold

        # check and handle class imbalance for classification tasks
        if task_type == 'classification' and args.use_smoteenn:
            print(f"\n{fold + 1} fold - checking class distribution:")
            if check_class_imbalance(y_train_fold, threshold=args.imbalance_threshold):
                print("detected class imbalance, applying SMOTEENN...")
                X_train_fold, y_train_fold = apply_smoteenn(
                    X_train_fold, y_train_fold)
                print(
                    f"SMOTEENN processed training set size: {X_train_fold.shape}")

                y_train_fold_scaled = y_train_fold.astype(np.float32)
            else:
                print("class distribution is relatively balanced, not using SMOTEENN")
        elif task_type == 'classification':
            print(f"\n{fold + 1} fold - class distribution:")
            unique, counts = np.unique(y_train_fold, return_counts=True)
            print(
                f"training set class distribution: {dict(zip(unique, counts))}")
            print("SMOTEENN not enabled (use --use_smoteenn to enable)")

        train_loader, val_loader = prep_dataloader(
            X_train_fold, y_train_fold_scaled,
            X_val_fold, y_val_fold_scaled,
            dataset_config.training_config.batch_size,
            label_inverse_transform=label_inverse_transform
        )

        input_dim = X_train_fold.shape[1]
        if model_type == 'MLP':
            model = NeuralNet(
                input_dim=input_dim,
                num_classes=1,
                hidden_dims=dataset_config.model_config.hidden_dims,
                dropout_rates=dataset_config.model_config.dropout_rates,
                activation=dataset_config.model_config.activation,
                use_normalization=dataset_config.model_config.use_normalization,
                task_type=dataset_config.task_type,
                use_input_layernorm=args.use_input_layernorm,
            ).to(device)
        elif model_type == 'MLP+TL':
            model = TransformerMLPModel(
                input_dim=input_dim,
                num_heads=dataset_config.model_config.num_heads,
                transformer_dim=dataset_config.model_config.transformer_dim,
                num_layers=dataset_config.model_config.num_layers,
                seq_length=input_dim,
                num_classes=1,
                hidden_dims=dataset_config.model_config.hidden_dims,
                dropout_rates=dataset_config.model_config.dropout_rates,
                task_type=dataset_config.task_type,
                transformer_dropout=dataset_config.model_config.transformer_dropout,
            ).to(device)
        else:
            raise ValueError(f"unsupported model type: {model_type}")

        model.apply(init_weights)

        class TrainingArgs:
            def __init__(self, training_config, use_warmup=False):
                self.lr = training_config.learning_rate
                self.decay = training_config.weight_decay
                self.epochs = training_config.epochs
                self.patience = training_config.patience
                self.use_warmup = use_warmup

        training_args = TrainingArgs(
            dataset_config.training_config, use_warmup=args.use_warmup)

        if task_type == 'classification':
            trained_model = train_kfold_classification_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                args=training_args,
                device=device,
                fold=fold,
                dataset_name=dataset_name,
                fingerprint_name=fingerprint_type,
                model_name=model_type,
            )

            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_test_fold),
                              torch.FloatTensor(y_test_fold)),
                batch_size=dataset_config.training_config.batch_size,
                shuffle=False
            )

            trained_model.eval()
            test_preds = []
            test_labels = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = torch.sigmoid(trained_model(X_batch))
                    test_preds.extend(outputs.cpu().numpy())
                    test_labels.extend(y_batch.cpu().numpy())

            test_preds = np.array(test_preds).flatten()
            test_labels = np.array(test_labels).flatten()

            test_predictions_df = pd.DataFrame({
                'true_labels': test_labels,
                'predicted_proba': test_preds
            })
            test_predictions_file = os.path.join(
                args.results_path,
                dataset_name, model_type, "kfold", f'{dataset_name}_{fingerprint_type}_{args.target_assay}_{model_type}_fold{fold+1}_test.csv'
            )
            os.makedirs(os.path.dirname(test_predictions_file), exist_ok=True)
            test_predictions_df.to_csv(test_predictions_file, index=False)
            print(f"test set predictions saved to: {test_predictions_file}")

        else:  # regression
            trained_model = train_kfold_regression_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                args=training_args,
                device=device,
                fold=fold,
                dataset_name=dataset_name,
                fingerprint_name=fingerprint_type,
                model_name=model_type,
            )

            y_test_fold_scaled = y_test_fold
            if task_type == 'regression' and dataset_name in ['ESOL'] and 'scaler' in locals():
                y_test_fold_scaled = scaler.transform(
                    y_test_fold.reshape(-1, 1)).astype(np.float32)

            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_test_fold),
                              torch.FloatTensor(y_test_fold_scaled)),
                batch_size=dataset_config.training_config.batch_size,
                shuffle=False
            )

            trained_model.eval()
            test_preds = []
            test_labels = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = trained_model(X_batch)
                    test_preds.extend(outputs.cpu().numpy())
                    test_labels.extend(y_batch.cpu().numpy())

            test_preds = np.array(test_preds).flatten()
            test_labels = np.array(test_labels).flatten()

            if task_type == 'regression' and dataset_name in ['ESOL'] and 'scaler' in locals():
                test_preds_original = scaler.inverse_transform(
                    test_preds.reshape(-1, 1)).flatten()
                test_labels_original = scaler.inverse_transform(
                    test_labels.reshape(-1, 1)).flatten()
            else:
                test_preds_original = test_preds
                test_labels_original = test_labels

            test_predictions_df = pd.DataFrame({
                'true_values': test_labels_original,
                'predicted_values': test_preds_original
            })
            test_predictions_file = os.path.join(
                args.results_path,
                dataset_name, model_type, "kfold",
                f'{dataset_name}_{fingerprint_type}_{args.target_assay}_{model_type}_fold{fold+1}_test.csv'
            )
            os.makedirs(os.path.dirname(test_predictions_file), exist_ok=True)
            test_predictions_df.to_csv(test_predictions_file, index=False)
            print(f"test set predictions saved to: {test_predictions_file}")

    # record experiment results to CSV
    experiment_metrics = {
        'fingerprint_type': fingerprint_type,
        'target_assay': args.target_assay,
        'model_type': model_type,
        'num_folds': dataset_config.training_config.n_splits,
        'status': 'completed',
        'input_dim': X_features.shape[1] if 'X_features' in locals() else None,
        'total_samples': len(X_features) if 'X_features' in locals() else None
    }

    output_file = os.path.join(
        args.results_path, dataset_name, model_type, "configs", "experiment_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    log_experiment_result(
        dataset=dataset_name,
        model_config=dataset_config.model_config,
        training_config=dataset_config.training_config,
        metrics=experiment_metrics,
        timestamp=timestamp,
        output_file=output_file
    )
    print(f"experiment results recorded to: {output_file}")

    return {
        'dataset': dataset_name,
        'fingerprint': fingerprint_type,
        'target_assay': args.target_assay,
        'model_type': model_type,
        'num_folds': dataset_config.training_config.n_splits,
        'status': 'completed'
    }


def run_single_experiment(dataset_name, fingerprint_type, model_type, args, device):
    """run feature network experiment (imputed + fingerprints + other assay features)"""

    print(
        f"Running Feature Network experiment: {fingerprint_type} - {args.target_assay} - {model_type}\n")

    if not os.path.exists(args.multi_assay_data):
        print(
            f"error: multi-assay data file not found: {args.multi_assay_data}")
        return None

    df_raw = pd.read_csv(args.multi_assay_data)
    if 'Identifier' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Identifier'])

    if args.target_assay not in df_raw.columns:
        print(f"error: target assay '{args.target_assay}' not found in data")
        return None

    # record experiment configuration
    dataset_config, fingerprint_config = get_model_config(
        dataset_name, fingerprint_type, model_type)
    if getattr(dataset_config, 'requires_imputation', True):
        print("imputation required")
        base_dir = os.path.dirname(__file__)
        imputed_data_dir = os.path.join(
            base_dir, 'src', 'data', 'Imputed_data')
        save_path = os.path.join(
            imputed_data_dir, f'{dataset_name}_{fingerprint_type}_imputed.csv')
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"imputed data file not found: {save_path}. please run imputation process to generate this file, then try again."
            )
        df_imputed = pd.read_csv(save_path)

        if getattr(dataset_config, 'use_feature_net', True):
            print("using feature network")
            X_features, y_target, available_indices = prepare_feature_network_data(
                df_imputed, args.target_assay, fingerprint_type, smiles_column=args.id_col, dataset_name=dataset_name)
            df_used = df_imputed
        else:
            print("not using feature network")
            X_features, y_target, available_indices = prepare_fingerprint_only_data(
                df_imputed, args.target_assay, fingerprint_type, smiles_column=args.id_col, dataset_name=dataset_name)
            df_used = df_imputed
    else:
        print("no imputation required")
        # 不需要補值，直接使用原始資料
        if getattr(dataset_config, 'use_feature_net', True):
            print("using feature network")
            X_features, y_target, available_indices = prepare_feature_network_data(
                df_raw, args.target_assay, fingerprint_type, smiles_column=args.id_col, dataset_name=dataset_name)
            df_used = df_raw
        else:
            print("not using feature network")
            X_features, y_target, available_indices = prepare_fingerprint_only_data(
                df_raw, args.target_assay, fingerprint_type, smiles_column=args.id_col, dataset_name=dataset_name)
            df_used = df_raw

    # data splitting: determine splitting method based on whether test_id is specified
    task_type = dataset_config.task_type

    if args.test_id is not None:
        # case 1: specify single SMILES as test set, remaining data 8:2 for train/val
        print(f"\nspecify single SMILES as test set: {args.test_id}")

        if args.id_col not in df_used.columns:
            raise ValueError(f"id_col '{args.id_col}' not found in df_used.")

        raw_hits = df_used.index[df_used[args.id_col] == args.test_id]
        if len(raw_hits) == 0:
            raise KeyError(
                f"no row found for {args.id_col} == {args.test_id} in df_used.")
        if len(raw_hits) > 1:
            raise ValueError(
                f"{args.id_col} == {args.test_id} not unique (found {len(raw_hits)} rows in df_used).")
        test_raw_idx = raw_hits[0]

        av_idx = pd.Index(available_indices)
        try:
            test_pos = int(av_idx.get_loc(test_raw_idx))
        except KeyError:
            raise KeyError(
                f"specified test row (df index {test_raw_idx}) not found in available_indices.\n"
                "possible reasons: the row was filtered out during preprocessing (e.g. SMILES missing/invalid)."
            )

        n = len(av_idx)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_pos] = False

        X_te = X_features[[test_pos]]
        y_te = y_target[[test_pos]]

        X_remain = X_features[train_mask]
        y_remain = y_target[train_mask]

        if task_type == 'classification':
            idx_train, idx_val = train_test_split(
                np.arange(len(X_remain)), test_size=0.2, stratify=y_remain, random_state=42
            )
        else:
            idx_train, idx_val = train_test_split(
                np.arange(len(X_remain)), test_size=0.2, random_state=42
            )

        X_tr = X_remain[idx_train]
        y_tr = y_remain[idx_train]
        X_val = X_remain[idx_val]
        y_val = y_remain[idx_val]

        print(f"  training set: {len(X_tr)} samples")
        print(f"  validation set: {len(X_val)} samples")
        print(f"  test set: {len(X_te)} samples")

    else:
        # case 2: no test set specified, all data split 80:20 for train/val
        print(f"\nno test set specified, split all data 80:20 for train/val")

        if task_type == 'classification':
            idx_train, idx_val = train_test_split(
                np.arange(len(X_features)), test_size=0.2, stratify=y_target, random_state=42
            )
        else:
            idx_train, idx_val = train_test_split(
                np.arange(len(X_features)), test_size=0.2, random_state=42
            )

        X_tr = X_features[idx_train]
        y_tr = y_target[idx_train]
        X_val = X_features[idx_val]
        y_val = y_target[idx_val]

        # no test set
        X_te = None
        y_te = None

        print(f"  training set: {len(X_tr)} samples")
        print(f"  validation set: {len(X_val)} samples")
        print(f"  test set: none")

    # only standardize y for regression tasks and dataset is ESOL (to avoid data leakage: fit on current fold training subset)
    label_inverse_transform = None
    if task_type == 'regression' and dataset_name in ['ESOL']:
        scaler = StandardScaler()
        y_tr_scaled = scaler.fit_transform(
            y_tr.reshape(-1, 1)).astype(np.float32)
        y_val_scaled = scaler.transform(
            y_val.reshape(-1, 1)).astype(np.float32)
        if args.test_id is not None:
            y_te_scaled = scaler.transform(
                y_te.reshape(-1, 1)).astype(np.float32)
        else:
            y_te_scaled = None
        label_inverse_transform = scaler.inverse_transform
    else:
        y_tr_scaled = y_tr
        y_val_scaled = y_val
        y_te_scaled = y_te if args.test_id is not None else None

    # check and handle class imbalance for classification tasks
    if task_type == 'classification' and args.use_smoteenn:
        if check_class_imbalance(y_tr, threshold=args.imbalance_threshold):
            print("detected class imbalance, applying SMOTEENN...")
            X_tr, y_tr = apply_smoteenn(X_tr, y_tr)
            print(f"SMOTEENN processed training set size: {X_tr.shape}")
            # synchronize updated scaled labels (classification tasks do not scale, but need to maintain size consistency and correct dtype)
            y_tr_scaled = y_tr.astype(np.float32)
        else:
            print("class distribution is relatively balanced, not using SMOTEENN")
    elif task_type == 'classification':
        unique, counts = np.unique(y_tr, return_counts=True)
        print(f"training set class distribution: {dict(zip(unique, counts))}")
        print("SMOTEENN not enabled (use --use_smoteenn to enable)")

    train_loader, val_loader = prep_dataloader(
        X_tr, y_tr_scaled,
        X_val, y_val_scaled,
        dataset_config.training_config.batch_size,
        label_inverse_transform=label_inverse_transform
    )

    input_dim = X_tr.shape[1]
    if model_type == 'MLP':
        model = NeuralNet(
            input_dim=input_dim,
            num_classes=1,
            hidden_dims=dataset_config.model_config.hidden_dims,
            dropout_rates=dataset_config.model_config.dropout_rates,
            activation=dataset_config.model_config.activation,
            use_normalization=dataset_config.model_config.use_normalization,
            task_type=dataset_config.task_type,
            use_input_layernorm=args.use_input_layernorm,
        ).to(device)
    elif model_type == 'MLP+TL':
        model = TransformerMLPModel(
            input_dim=input_dim,
            num_heads=dataset_config.model_config.num_heads,
            transformer_dim=dataset_config.model_config.transformer_dim,
            num_layers=dataset_config.model_config.num_layers,
            seq_length=input_dim,
            num_classes=1,
            hidden_dims=dataset_config.model_config.hidden_dims,
            dropout_rates=dataset_config.model_config.dropout_rates,
            task_type=dataset_config.task_type,
            return_attn_weights=False,
            transformer_dropout=dataset_config.model_config.transformer_dropout,
        ).to(device)
    else:
        raise ValueError(f"unsupported model type: {model_type}")

    model.apply(init_weights)

    class TrainingArgs:
        def __init__(self, training_config, use_warmup=False):
            self.lr = training_config.learning_rate
            self.decay = training_config.weight_decay
            self.epochs = training_config.epochs
            self.patience = training_config.patience
            self.use_warmup = use_warmup

    training_args = TrainingArgs(
        dataset_config.training_config, use_warmup=args.use_warmup)

    if task_type == 'classification':
        trained_model = train_single_fold_classification_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            args=training_args,
            device=device,
            dataset_name=dataset_name,
            fingerprint_name=fingerprint_type,
            model_name=model_type,
        )
        if args.test_id is None:
            model_save_dir = os.path.join(
                args.results_path, dataset_name, model_type, "models")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(
                model_save_dir,
                f'{dataset_name}_{fingerprint_type}_{args.target_assay}_{model_type}_model.pth'
            )
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_type': model_type,
                'fingerprint_type': fingerprint_type,
                'dataset_name': dataset_name,
                'target_assay': args.target_assay,
                'task_type': task_type,
                'input_dim': input_dim,
                'hidden_dims': dataset_config.model_config.hidden_dims,
                'dropout_rates': dataset_config.model_config.dropout_rates,
                'activation': dataset_config.model_config.activation,
                'use_normalization': dataset_config.model_config.use_normalization,
                'use_input_layernorm': args.use_input_layernorm,
                # TransformerMLPModel specific parameters (None for MLP)
                'num_heads': dataset_config.model_config.num_heads,
                'transformer_dim': dataset_config.model_config.transformer_dim,
                'num_layers': dataset_config.model_config.num_layers,
                'transformer_dropout': dataset_config.model_config.transformer_dropout,
                # classification tasks have no scaler
                'label_scaler': None,
            }, model_save_path)
            print(f"model weights saved to: {model_save_path}")

        # only evaluate on test set if test_id is specified
        else:
            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_te),
                              torch.FloatTensor(y_te_scaled)),
                batch_size=1,
                shuffle=False
            )

            trained_model.eval()
            test_preds = []
            test_labels = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = trained_model(X_batch)
                    test_preds.extend(outputs.cpu().numpy())
                    test_labels.extend(y_batch.cpu().numpy())

            test_preds = np.array(test_preds).flatten()
            test_labels = np.array(test_labels).flatten()

            test_predictions_df = pd.DataFrame({
                'true_values': test_labels,
                'predicted_values': test_preds
            })

            test_predictions_file = os.path.join(
                args.results_path, dataset_name, model_type, "single_mol", f'{dataset_name}_{fingerprint_type}_{args.target_assay}_{model_type}_test.csv')
            os.makedirs(os.path.dirname(test_predictions_file), exist_ok=True)
            test_predictions_df.to_csv(test_predictions_file, index=False)
            print(f"test set predictions saved to: {test_predictions_file}")

            # if single SMILES is specified and MLP+TL is used, output the last layer head-avg attention heatmap
            if model_type == 'MLP+TL':
                try:
                    export_single_smiles_attention(
                        trained_model, args, df_used, available_indices, X_features,
                        dataset_name, fingerprint_type, device
                    )
                except Exception as e:
                    print(f"[attention] error occurred during generation: {e}")

    else:
        trained_model = train_single_fold_regression_model(train_loader=train_loader, val_loader=val_loader, model=model, args=training_args,
                                                           device=device, dataset_name=dataset_name, fingerprint_name=fingerprint_type, model_name=model_type)
        if args.test_id is None:
            model_save_dir = os.path.join(
                args.results_path, dataset_name, model_type, "models")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(
                model_save_dir,
                f'{dataset_name}_{fingerprint_type}_{args.target_assay}_{model_type}_model.pth'
            )
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_type': model_type,
                'fingerprint_type': fingerprint_type,
                'dataset_name': dataset_name,
                'target_assay': args.target_assay,
                'task_type': task_type,
                'input_dim': input_dim,
                'hidden_dims': dataset_config.model_config.hidden_dims,
                'dropout_rates': dataset_config.model_config.dropout_rates,
                'activation': dataset_config.model_config.activation,
                'use_normalization': dataset_config.model_config.use_normalization,
                'use_input_layernorm': args.use_input_layernorm,
                # TransformerMLPModel specific parameters (None for MLP)
                'num_heads': dataset_config.model_config.num_heads,
                'transformer_dim': dataset_config.model_config.transformer_dim,
                'num_layers': dataset_config.model_config.num_layers,
                'transformer_dropout': dataset_config.model_config.transformer_dropout,
                # regression tasks have no scaler
                'label_scaler': scaler if label_inverse_transform is not None else None,
            }, model_save_path)
            print(f"model weights saved to: {model_save_path}")

        # only evaluate on test set if test_id is specified
        else:
            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_te),
                              torch.FloatTensor(y_te_scaled)),
                batch_size=1,
                shuffle=False
            )

            trained_model.eval()
            test_preds = []
            test_labels = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = trained_model(X_batch)
                    test_preds.extend(outputs.cpu().numpy())
                    test_labels.extend(y_batch.cpu().numpy())

            test_preds = np.array(test_preds).flatten()
            test_labels = np.array(test_labels).flatten()

            if label_inverse_transform is not None:
                test_preds_original = label_inverse_transform(
                    test_preds.reshape(-1, 1)).flatten()
                test_labels_original = label_inverse_transform(
                    test_labels.reshape(-1, 1)).flatten()
            else:
                test_preds_original = test_preds
                test_labels_original = test_labels

            test_predictions_df = pd.DataFrame({
                'true_values': test_labels_original,
                'predicted_values': test_preds_original
            })
            test_predictions_file = os.path.join(
                args.results_path, dataset_name, model_type, "single_mol", f'{dataset_name}_{fingerprint_type}_{args.target_assay}_{args.test_id}.csv')
            os.makedirs(os.path.dirname(test_predictions_file), exist_ok=True)
            test_predictions_df.to_csv(test_predictions_file, index=False)
            print(f"test set predictions saved to: {test_predictions_file}")

            # if single SMILES is specified and MLP+TL is used, output the last layer head-avg attention heatmap
            if model_type == 'MLP+TL':
                try:
                    export_single_smiles_attention(
                        trained_model, args, df_used, available_indices, X_features,
                        dataset_name, fingerprint_type, device
                    )
                except Exception as e:
                    print(f"[attention] error occurred during generation: {e}")

    return {
        'dataset': dataset_name,
        'fingerprint': fingerprint_type,
        'target_assay': args.target_assay,
        'model_type': model_type,
        'num_folds': dataset_config.training_config.n_splits,
        'status': 'completed'
    }


def main():
    args = parse_arguments()
    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    fingerprint_type = args.fingerprint
    print(f"Using fingerprint type: {fingerprint_type}")

    if args.use_single_fold:
        run_single_experiment(
            args.dataset, fingerprint_type, args.model_type, args, device)
    else:
        run_kfold_experiment(
            args.dataset, fingerprint_type, args.model_type, args, device)


if __name__ == '__main__':
    main()
