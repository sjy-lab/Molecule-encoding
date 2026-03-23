# Output files:
#   1. ROC: {dataset}_{fingerprint}_{model}_test_folds_roc.png
#   2. fold details: {dataset}_{fingerprint}_{model}_test_fold_details.csv
#   3. fold averaged results: fold_averaged_results/{dataset}_{fingerprint}_{model}_test_fold_averaged_metrics.csv
#   5. consolidated comparison table: consolidated_results/{dataset}_{model}_fold_averaged_comparison.csv

import random
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, roc_curve, auc, precision_recall_curve
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import argparse
np.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BACE')
parser.add_argument('--model', type=str, default='MLP')
args = parser.parse_args()

dataset = args.dataset
model = args.model
results_path = f'experiments/{dataset}/{model}/kfold'
output_path = f'experiments/{dataset}/{model}/kfold/result_test'


def load_predictions(file_path):

    df = pd.read_csv(file_path)

    return df.copy()

def parse_filename(filename):
    """
    format: {dataset}_{fingerprint}_{target_assay}_{model}_fold{n}_test.csv
    """
    base_name = filename.replace('.csv', '')

    fold_pattern = r'_fold(\d+)_test$'
    fold_match = re.search(fold_pattern, base_name)
    if not fold_match:
        return None, None, None, None, None
    
    fold = int(fold_match.group(1))
    name_without_fold = base_name[:fold_match.start()]
    
    parts = name_without_fold.split('_')
    if len(parts) < 4:
        return None, None, None, None, None
    
    dataset = parts[0]
    fingerprint = parts[1]
    model = parts[-1]

    target_assay = '_'.join(parts[2:-1]) if len(parts) > 3 else None
    
    return dataset, fingerprint, model, fold, target_assay


def evaluate_classification_with_folds(combined_df, dataset, fingerprint, model, target_assay=None):

    plt.figure(figsize=(12, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    fold_results = []
    unique_folds = sorted(combined_df['fold'].unique())
    print(f"Processing folds in order: {unique_folds}")

    for fold in unique_folds:
        fold_data = combined_df[combined_df['fold'] == fold]

        true_labels = fold_data['true_labels'].values.astype(int)
        pred_probas = fold_data['predicted_proba'].values

        test_preds = (pred_probas >= 0.5).astype(int)
        f1 = f1_score(true_labels, test_preds)
        precision = precision_score(true_labels, test_preds)
        recall = recall_score(true_labels, test_preds)
        accuracy = accuracy_score(true_labels, test_preds)

        try:
            fpr, tpr, _ = roc_curve(true_labels, pred_probas)
            fold_auc = auc(fpr, tpr)

            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(fold_auc)

            plt.plot(fpr, tpr, lw=1, alpha=0.6,
                     label=f'Fold {fold} (AUC = {fold_auc:.3f})')

            fold_results.append({
                'fold': fold,
                'f1': round(f1, 3),
                'recall': round(recall, 3),
                'precision': round(precision, 3),
                'accuracy': round(accuracy, 3),
                'auc': round(fold_auc, 3),
            })

        except ValueError:
            print(f"Warning: Could not calculate AUC for fold {fold}")
            fold_results.append({
                'fold': fold,
                'f1': round(f1, 3),
                'recall': round(recall, 3),
                'precision': round(precision, 3),
                'accuracy': round(accuracy, 3),
                'auc': 0.0,
            })

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})', lw=3)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             label='Random Classifier (AUC = 0.50)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    title = f'{dataset} - {fingerprint} - {model}\nTest Set ROC Curves for Each Fold'
    file_prefix = f'{dataset}_{fingerprint}_{model}'

    plt.title(title)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    os.makedirs(output_path, exist_ok=True)
    roc_save_path = os.path.join(
        output_path, f'{file_prefix}_test_folds_roc.png')
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Test set multi-fold ROC curve saved to {roc_save_path}")


    if fold_results:
        fold_f1s = [result['f1'] for result in fold_results]
        fold_precisions = [result['precision'] for result in fold_results]
        fold_recalls = [result['recall'] for result in fold_results]
        fold_accuracies = [result['accuracy'] for result in fold_results]
        fold_aucs = [result['auc']
                     for result in fold_results if result['auc'] > 0]

        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        mean_precision = np.mean(fold_precisions)
        std_precision = np.std(fold_precisions)
        mean_recall = np.mean(fold_recalls)
        std_recall = np.std(fold_recalls)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
        else:
            mean_auc = 0.0
            std_auc = 0.0

    fold_df = pd.DataFrame(fold_results)
    fold_save_path = os.path.join(
        output_path, f'{file_prefix}_test_fold_details.csv')
    fold_df.to_csv(fold_save_path, index=False)
    print(f"Test fold details saved to {fold_save_path}")

    avg_results_folder = os.path.join(output_path, 'fold_averaged_results')
    os.makedirs(avg_results_folder, exist_ok=True)

    avg_results_data = {
        'metric': ['f1', 'precision', 'recall', 'accuracy', 'auc'],
        'formatted': [
            f"{mean_f1:.3f} ± {std_f1:.3f}",
            f"{mean_precision:.3f} ± {std_precision:.3f}",
            f"{mean_recall:.3f} ± {std_recall:.3f}",
            f"{mean_accuracy:.3f} ± {std_accuracy:.3f}",
            f"{mean_auc:.3f} ± {std_auc:.3f}"
        ]
    }

    avg_results_df = pd.DataFrame(avg_results_data)
    avg_save_path = os.path.join(
        avg_results_folder, f'{file_prefix}_test_fold_averaged_metrics.csv')
    avg_results_df.to_csv(avg_save_path, index=False)
    print(f"Test fold averaged results saved to {avg_save_path}")


    return {
        'f1_mean': round(mean_f1, 3),
        'f1_std': round(std_f1, 3),
        'precision_mean': round(mean_precision, 3),
        'precision_std': round(std_precision, 3),
        'recall_mean': round(mean_recall, 3),
        'recall_std': round(std_recall, 3),
        'accuracy_mean': round(mean_accuracy, 3),
        'accuracy_std': round(std_accuracy, 3),
        'auc_mean': round(mean_auc, 3),
        'auc_std': round(std_auc, 3),
        'num_folds': len(fold_results)
    }

def evaluate_regression_with_folds(combined_df, dataset, fingerprint, model, target_assay=None):

    fold_results = []

    unique_folds = sorted(combined_df['fold'].unique())
    print(f"Processing folds in order: {unique_folds}")

    for fold in unique_folds:
        fold_data = combined_df[combined_df['fold'] == fold].copy()

        fold_data = fold_data.sort_index().reset_index(drop=True)

        if 'true_values' in fold_data.columns:
            true_values = fold_data['true_values'].values
        else:
            raise KeyError("regression evaluation missing true_values column")

        if 'predicted_values' in fold_data.columns:
            pred_values = fold_data['predicted_values'].values
        else:
            raise KeyError("regression evaluation missing predicted_values column")
        
        print(f"Fold {fold}: {len(true_values)} samples")
        print(
            f"  True values range: [{np.min(true_values):.4f}, {np.max(true_values):.4f}]")
        print(
            f"  Pred values range: [{np.min(pred_values):.4f}, {np.max(pred_values):.4f}]")
        print(f"  First 3 true: {true_values[:3]}")
        print(f"  First 3 pred: {pred_values[:3]}")

        mse = mean_squared_error(true_values, pred_values)
        rmse = mse ** 0.5
        mae = mean_absolute_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)

        print(
            f"  Metrics: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")

        fold_results.append({
            'fold': fold,
            'mse': round(mse, 3),
            'rmse': round(rmse, 3),
            'mae': round(mae, 3),
            'r2': round(r2, 3)
        })

    if fold_results:
        fold_mses = [result['mse'] for result in fold_results]
        fold_rmses = [result['rmse'] for result in fold_results]
        fold_maes = [result['mae'] for result in fold_results]
        fold_r2s = [result['r2'] for result in fold_results]

        mean_mse = np.mean(fold_mses)
        std_mse = np.std(fold_mses)
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)

    if 'true_values' in combined_df.columns:
        all_true_values = combined_df['true_values'].values
    else:
        raise KeyError("regression evaluation missing true_values column")

    if 'predicted_values' in combined_df.columns:
        all_pred_values = combined_df['predicted_values'].values
    else:
        raise KeyError("regression evaluation missing predicted_values column")

    overall_mse = mean_squared_error(all_true_values, all_pred_values)
    overall_rmse = overall_mse ** 0.5
    overall_mae = mean_absolute_error(all_true_values, all_pred_values)
    overall_r2 = r2_score(all_true_values, all_pred_values)

    if target_assay:
        file_prefix = f'{dataset}_{fingerprint}_{target_assay}_{model}'
    else:
        file_prefix = f'{dataset}_{fingerprint}_{model}'

    os.makedirs(output_path, exist_ok=True)

    fold_df = pd.DataFrame(fold_results)
    fold_save_path = os.path.join(
        output_path, f'{file_prefix}_test_fold_details.csv')
    fold_df.to_csv(fold_save_path, index=False)
    print(f"Test fold details saved to {fold_save_path}")

    avg_results_folder = os.path.join(output_path, 'fold_averaged_results')
    os.makedirs(avg_results_folder, exist_ok=True)

    avg_results_data = {
        'metric': ['mse', 'rmse', 'mae', 'r2'],
        'formatted': [
            f"{mean_mse:.3f} ± {std_mse:.3f}",
            f"{mean_rmse:.3f} ± {std_rmse:.3f}",
            f"{mean_mae:.3f} ± {std_mae:.3f}",
            f"{mean_r2:.3f} ± {std_r2:.3f}"
        ]
    }

    avg_results_df = pd.DataFrame(avg_results_data)

    avg_save_path = os.path.join(
        avg_results_folder, f'{file_prefix}_test_fold_averaged_metrics.csv')
    avg_results_df.to_csv(avg_save_path, index=False)
    print(f"Test fold averaged results saved to {avg_save_path}")


    return {
        'mse_mean': round(mean_mse, 3),
        'mse_std': round(std_mse, 3),
        'rmse_mean': round(mean_rmse, 3),
        'rmse_std': round(std_rmse, 3),
        'mae_mean': round(mean_mae, 3),
        'mae_std': round(std_mae, 3),
        'r2_mean': round(mean_r2, 3),
        'r2_std': round(std_r2, 3),

        'mse_combined': round(overall_mse, 3),
        'rmse_combined': round(overall_rmse, 3),
        'mae_combined': round(overall_mae, 3),
        'r2_combined': round(overall_r2, 3),

        'mse': round(overall_mse, 3),
        'rmse': round(overall_rmse, 3),
        'mae': round(overall_mae, 3),
        'r2': round(overall_r2, 3),

        'num_folds': len(fold_results)
    }

def consolidate_results_by_dataset_model():

    avg_results_folder = os.path.join(output_path, 'fold_averaged_results')

    if not os.path.exists(avg_results_folder):
        print("cannot find fold averaged results folder, skipping consolidation")
        return

    avg_files = [f for f in os.listdir(avg_results_folder) if f.endswith(
        '_test_fold_averaged_metrics.csv')]

    dataset_model_groups = defaultdict(lambda: {'avg': [], 'combined': []})

    for file in avg_files:
        # analyze file name: dataset_fingerprint_model_test_fold_averaged_metrics.csv
        # or dataset_fingerprint_target_model_test_fold_averaged_metrics.csv
        base_name = file.replace('_test_fold_averaged_metrics.csv', '')
        parts = base_name.split('_')

        if len(parts) >= 3:
            dataset = parts[0]
            fingerprint = parts[1]
            if len(parts) == 3:  # dataset_fingerprint_model
                model = parts[2]
                target_assay = None
            elif len(parts) == 4:  # dataset_fingerprint_target_model
                target_assay = parts[2]
                model = parts[3]
            else: 
                model = parts[-1]
                target_assay = '_'.join(
                    parts[2:-1]) if len(parts) > 3 else None

            key = (dataset, model, target_assay)
            dataset_model_groups[key]['avg'].append((fingerprint, file))

    consolidated_folder = os.path.join(output_path, 'consolidated_results')
    os.makedirs(consolidated_folder, exist_ok=True)

    for (dataset, model, target_assay), files in dataset_model_groups.items():

        if files['avg']:
            avg_consolidated_data = []
            for fingerprint, file in sorted(files['avg']):
                file_path = os.path.join(avg_results_folder, file)
                df = pd.read_csv(file_path)

              
                metrics_dict = {'fingerprint_type': fingerprint}
                for _, row in df.iterrows():
                    metrics_dict[row['metric']] = row['formatted']

                avg_consolidated_data.append(metrics_dict)

  
            if avg_consolidated_data:
                avg_consolidated_df = pd.DataFrame(avg_consolidated_data)

            
                base_columns = ['fingerprint_type']
                metric_columns = [
                    col for col in avg_consolidated_df.columns if col != 'fingerprint_type']
                avg_consolidated_df = avg_consolidated_df[base_columns + sorted(
                    metric_columns)]

                if target_assay:
                    avg_filename = f'{dataset}_{target_assay}_{model}_fold_averaged_comparison.csv'
                else:
                    avg_filename = f'{dataset}_{model}_fold_averaged_comparison.csv'

                avg_save_path = os.path.join(consolidated_folder, avg_filename)
                avg_consolidated_df.to_csv(avg_save_path, index=False)
                print(f"Fold averaged results saved to: {avg_save_path}")


    print(f"\nConsolidated results saved to: {consolidated_folder}")

def main():

    prediction_files = [f for f in os.listdir(results_path)
                        if f.endswith('_test.csv') and 'fold' in f and not f.endswith('_summary.csv')]

    if not prediction_files:
        print(f"No test prediction files found in {results_path}")
        return

    file_groups = defaultdict(list)

    for prediction_file in prediction_files:
        dataset, fingerprint, model, fold, target_assay = parse_filename(
            prediction_file)
        if dataset and fingerprint and model:
            key = (dataset, fingerprint, model, target_assay)
            file_groups[key].append((prediction_file, fold))

    all_results = []

    for (dataset, fingerprint, model, target_assay), files in file_groups.items():
        print(f"\n{'='*60}")

        print(
            f"Evaluating Test Set: {dataset} - {fingerprint} - {target_assay} - {model}")

        print(f"Found {len(files)} fold(s)")
        print(f"{'='*60}")

        combined_predictions = []

        sorted_files = sorted(files, key=lambda x: (x[1], x[0]))

        for prediction_file, fold in sorted_files:
            file_path = os.path.join(results_path, prediction_file)
            fold_predictions = load_predictions(file_path)
            fold_predictions['fold'] = fold
            fold_predictions['file_source'] = prediction_file 
            combined_predictions.append(fold_predictions)
            print(
                f"Loaded fold {fold}: {len(fold_predictions)} samples from {prediction_file}")
        
        combined_df = pd.concat(combined_predictions, ignore_index=True)
        print(f"Total combined samples: {len(combined_df)}")
        
        is_classification = 'true_labels' in combined_df.columns
        task_type = 'classification' if is_classification else 'regression'
        
        if is_classification:
            metrics = evaluate_classification_with_folds(
                combined_df, dataset, fingerprint, model, target_assay
            )
            
            result_summary = {
                'dataset': dataset,
                'fingerprint': fingerprint,
                'model': model,
                'target_assay': target_assay,
                'task_type': task_type,
                'num_folds': metrics['num_folds'],
                'total_samples': len(combined_df),
                'f1_mean': metrics['f1_mean'],
                'f1_std': metrics['f1_std'],
                'precision_mean': metrics['precision_mean'],
                'precision_std': metrics['precision_std'],
                'recall_mean': metrics['recall_mean'],
                'recall_std': metrics['recall_std'],
                'accuracy_mean': metrics['accuracy_mean'],
                'accuracy_std': metrics['accuracy_std'],
                'auc_mean': metrics['auc_mean'],
                'auc_std': metrics['auc_std'],
            }
            
        else:  # regression
            metrics = evaluate_regression_with_folds(
                combined_df, dataset, fingerprint, model, target_assay
            )

            result_summary = {
                'dataset': dataset,
                'fingerprint': fingerprint,
                'model': model,
                'target_assay': target_assay,
                'task_type': task_type,
                'num_folds': metrics['num_folds'],
                'total_samples': len(combined_df),
                'mse_mean': metrics['mse_mean'],
                'mse_std': metrics['mse_std'],
                'rmse_mean': metrics['rmse_mean'],
                'rmse_std': metrics['rmse_std'],
                'mae_mean': metrics['mae_mean'],
                'mae_std': metrics['mae_std'],
                'r2_mean': metrics['r2_mean'],
                'r2_std': metrics['r2_std'],
            }
        
        all_results.append(result_summary)

    if all_results:
        classification_results = []
        regression_results = []

        for result in all_results:
            if result['task_type'] == 'classification':

                clean_result = {
                    'dataset': result['dataset'],
                    'fingerprint': result['fingerprint'],
                    'model': result['model'],
                    'target_assay': result['target_assay'],
                    'num_folds': result['num_folds'],
                    'total_samples': result['total_samples'],
                    'fold_avg_f1': result['f1_mean'],
                    'fold_avg_f1_std': result['f1_std'],
                    'fold_avg_precision': result['precision_mean'],
                    'fold_avg_precision_std': result['precision_std'],
                    'fold_avg_recall': result['recall_mean'],
                    'fold_avg_recall_std': result['recall_std'],
                    'fold_avg_accuracy': result['accuracy_mean'],
                    'fold_avg_accuracy_std': result['accuracy_std'],
                    'fold_avg_auc': result['auc_mean'],
                    'fold_avg_auc_std': result['auc_std'],
                }
                classification_results.append(clean_result)

            elif result['task_type'] == 'regression':
                clean_result = {
                    'dataset': result['dataset'],
                    'fingerprint': result['fingerprint'],
                    'model': result['model'],
                    'target_assay': result['target_assay'],
                    'num_folds': result['num_folds'],
                    'total_samples': result['total_samples'],
                    'fold_avg_r2': result['r2_mean'],
                    'fold_avg_r2_std': result['r2_std'],
                    'fold_avg_rmse': result['rmse_mean'],
                    'fold_avg_rmse_std': result['rmse_std'],
                    'fold_avg_mae': result['mae_mean'],
                    'fold_avg_mae_std': result['mae_std'],
                    'fold_avg_mse': result['mse_mean'],
                    'fold_avg_mse_std': result['mse_std'],
                }
                regression_results.append(clean_result)

        os.makedirs(output_path, exist_ok=True)



        summary_df = pd.DataFrame(all_results)

        print("\nSummary of all test set evaluations:")
        for _, row in summary_df.iterrows():
            if row['target_assay']:
                print(
                    f"\n{row['dataset']} - {row['fingerprint']} - {row['target_assay']} - {row['model']}:")
            else:
                print(
                    f"\n{row['dataset']} - {row['fingerprint']} - {row['model']}:")

            if row['task_type'] == 'classification':

                print(f"    F1: {row['f1_mean']:.3f} ± {row['f1_std']:.3f}")
                print(f"    AUC: {row['auc_mean']:.3f} ± {row['auc_std']:.3f}")
                print(
                    f"    Precision: {row['precision_mean']:.3f} ± {row['precision_std']:.3f}")
                print(
                    f"    Recall: {row['recall_mean']:.3f} ± {row['recall_std']:.3f}")
            else:

                print(
                    f"    R²: {row['r2_mean']:.3f} ± {row['r2_std']:.3f}")
                print(
                    f"    RMSE: {row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}")
                print(
                    f"    MAE: {row['mae_mean']:.3f} ± {row['mae_std']:.3f}")

    consolidate_results_by_dataset_model()


def consolidate_only():

    consolidate_results_by_dataset_model()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--consolidate-only':
        consolidate_only()
    else:
        main()
