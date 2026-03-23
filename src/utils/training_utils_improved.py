import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve, auc, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get appropriate device (GPU or CPU)"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curves(train_losses, val_losses, fold, dataset_name, fingerprint_name, model_name, use_single_fold=False, task_type=None):
    """
    Plot and save learning curves

    Args:
        train_losses: training loss list
        val_losses: validation loss list
        fold: current fold number
        dataset_name: dataset name
        fingerprint_name: fingerprint name
        model_name: model name
        task_type: task type ('classification' or 'regression')
    """

    plt.figure(figsize=(10, 6))

    if use_single_fold:
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-',
                 label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-',
                 label='Validation Loss', linewidth=2)

        min_val_loss_epoch = np.argmin(val_losses) + 1
        min_val_loss = min(val_losses)
        plt.scatter(min_val_loss_epoch, min_val_loss,
                    color='red', s=100, zorder=5)
        plt.annotate(f'Best Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch}',
                     xy=(min_val_loss_epoch, min_val_loss), xytext=(10, 10),
                     textcoords='offset points', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.title(
            f'Learning Curves - {dataset_name} - {fingerprint_name} - {model_name}\n({task_type.capitalize()}) - Single Fold', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)

        if task_type == 'classification':
            plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
        else:
            plt.ylabel('Mean Squared Error Loss', fontsize=12)

        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # set y-axis from 0 (if minimum value is not negative)
        y_min = min(min(train_losses), min(val_losses))
        if y_min >= 0:
            plt.ylim(bottom=0)

        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        plt.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}\nTotal Epochs: {len(train_losses)}',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        filename = f'{dataset_name}_{fingerprint_name}_{model_name}_learning_curve.png'
        filepath = os.path.join(
            "experiments", dataset_name, model_name, "learning_curves", filename
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-',
                 label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-',
                 label='Validation Loss', linewidth=2)

        min_val_loss_epoch = np.argmin(val_losses) + 1
        min_val_loss = min(val_losses)
        plt.scatter(min_val_loss_epoch, min_val_loss,
                    color='red', s=100, zorder=5)
        plt.annotate(f'Best Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch}',
                     xy=(min_val_loss_epoch, min_val_loss), xytext=(10, 10),
                     textcoords='offset points', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.title(
            f'Learning Curves - {dataset_name} - {fingerprint_name} - {model_name}\nFold {fold+1} ({task_type.capitalize()})', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)

        if task_type == 'classification':
            plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
        else:
            plt.ylabel('Mean Squared Error Loss', fontsize=12)

        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        y_min = min(min(train_losses), min(val_losses))
        if y_min >= 0:
            plt.ylim(bottom=0)

        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        plt.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}\nTotal Epochs: {len(train_losses)}',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        filename = f'{dataset_name}_{fingerprint_name}_{model_name}_fold{fold+1}_learning_curve.png'
        filepath = os.path.join(
            "experiments", dataset_name, model_name, "learning_curves", filename
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Learning curves saved to: {filepath}")


def prep_dataloader(X_train, y_train, X_test, y_test, batch_size, label_inverse_transform=None, single_test=False):
    """Prepare DataLoader"""
    # ensure input is numpy.ndarray
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    # --- test set only one sample shape correction/check ---
    if single_test:
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(1, -1)
        if X_test.shape[0] != 1:
            raise ValueError(
                f"Test set should only contain 1 sample, but currently is {X_test.shape[0]} samples.\n"
                "Please use split_one_as_test_by_id() to select one sample first.\n"
            )
        test_batch_size = 1
        print(
            f"Test set only contains 1 sample, converted to {X_test.shape[0]} samples.")
    else:
        test_batch_size = batch_size

    if label_inverse_transform is not None:
        class _DatasetWithInverse(TensorDataset):
            def __init__(self, *tensors):
                super().__init__(*tensors)
                self._label_inverse_transform = label_inverse_transform

            def inverse_transform_labels(self, y):
                if self._label_inverse_transform is None:
                    return y
                arr = y
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                arr = np.array(arr).reshape(-1, 1)
                return self._label_inverse_transform(arr)

        train_dataset = _DatasetWithInverse(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = _DatasetWithInverse(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    else:
        train_dataset = TensorDataset(torch.FloatTensor(
            X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,  # one SMILES -> 1
                             shuffle=False)

    return train_loader, test_loader

def export_single_smiles_attention(trained_model, args, df_used, available_indices, X_features,
                                   dataset_name, fingerprint_type, device):
    trained_model.return_attn_weights = True
    trained_model.eval()

    sm_col = args.id_col
    mask = (df_used[sm_col] == args.test_id)
    if not mask.any():
        print(f"can not find SMILES: {args.test_id}")
        return

    original_idx = df_used.index[mask][0]
    if original_idx not in available_indices:
        print(f"original_idx not in available_indices: {args.test_id}")
        return

    pos = available_indices.index(original_idx)
    x_sample = X_features[pos]

    x_tensor = torch.from_numpy(x_sample).float().unsqueeze(0).to(device)

    with torch.no_grad():
        if getattr(args, 'verify_transformer', False):
            out, last_headavg, attn_all_layers, embedded, dbg = trained_model(
                x_tensor, return_embedded=True, return_debug=True)
        else:
            out, last_headavg, attn_all_layers, embedded = trained_model(
                x_tensor, return_embedded=True)

    # # (1, H, L, L) = (1, 8, 167, 167)
    # print(f"last_headavg.shape: {last_headavg.shape}")
    # # (H, L, L) = (8, 167, 167)
    # print(f"last_headavg[0].shape: {last_headavg[0].shape}")
    # # (L, L) = (167, 167)
    # print(f"last_headavg[0][0].shape: {last_headavg[0][0].shape}")

    attn_matrix = last_headavg[0].detach().cpu().numpy()

    safe_smiles = args.test_id.replace(
        '/', '_').replace('\\', '_').replace(' ', '_')
    heatmap_path = os.path.join(
        args.results_path, dataset_name, args.model_type, "heatmap",
        f'Morphine_{dataset_name}_{fingerprint_type}_{args.target_assay}_attn_{safe_smiles}.png')
    matrix_csv_path = os.path.join(
        args.results_path, dataset_name, args.model_type, "attention_matrix",
        f'Morphine_{dataset_name}_{fingerprint_type}_{args.target_assay}_attn_{safe_smiles}.csv')
    embedded_csv_path = os.path.join(
        args.results_path, dataset_name, args.model_type, "embedded_vector",
        f'Morphine_{dataset_name}_{fingerprint_type}_{args.target_assay}_embedded_{safe_smiles}.csv')

    # export attention heatmap
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_matrix, robust=True, cmap='YlGnBu')

    plt.title(f'Last-layer attention (head-avg)\nSMILES: {args.test_id}',
              fontsize=10)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # export attention matrix
    # (H, L, L) = (8, 167, 167)
    # print(f"attn_matrix.shape: {attn_matrix.shape}")
    os.makedirs(os.path.dirname(matrix_csv_path), exist_ok=True)
    pd.DataFrame(attn_matrix).to_csv(matrix_csv_path, index=False)

    # export embedded vector (onehot encoded)
    # (1, L, embedding_dim) = (1, 167, 167)
    # print(f"embedded.shape: {embedded.shape}")
    os.makedirs(os.path.dirname(embedded_csv_path), exist_ok=True)
    pd.DataFrame(embedded[0].detach().cpu().numpy()).to_csv(
        embedded_csv_path, index=False)

    print(f"Attention heatmap saved to: {heatmap_path}")
    print(f"Attention matrix saved to: {matrix_csv_path}")
    print(f"Embedded vector saved to: {embedded_csv_path}")

    # calculate and export the top 20 most attention key positions
    # attn_matrix shape: (seq_length, seq_length)
    # each row is the attention score of a query to all keys
    # sum along the query dimension (axis=0), get the total attention score for each key position
    total_attention_per_key = attn_matrix.sum(axis=0)  # shape: (seq_length,)

    # find the top 20 most attention key positions
    # if the sequence length is less than 20, take all
    top_k = min(20, len(total_attention_per_key))
    top_indices = np.argsort(total_attention_per_key)[
        ::-1][:top_k]  # descending order
    top_scores = total_attention_per_key[top_indices]

    # save as CSV
    top_keys_df = pd.DataFrame({
        'Key_Position': top_indices,
        'Total_Attention_Score': top_scores,
        'Rank': range(1, top_k + 1)
    })

    top_keys_csv_path = os.path.join(
        args.results_path, dataset_name, args.model_type, "top_keys",
        f'{dataset_name}_{fingerprint_type}_{args.target_assay}_top20keys_{safe_smiles}.csv')
    os.makedirs(os.path.dirname(top_keys_csv_path), exist_ok=True)
    top_keys_df.to_csv(top_keys_csv_path, index=False)

    print(
        f"Top {top_k} most attention key positions (CSV) saved to: {top_keys_csv_path}")
    print(
        f"Top 10: {list(top_indices[:10])} (positions), scores: {[f'{s:.4f}' for s in top_scores[:5]]}")

    # # debug output (optional)
    # if getattr(args, 'verify_transformer', False):
    #     debug_dir = os.path.join(
    #         args.results_path, dataset_name, args.model_type, "debug_layers")
    #     os.makedirs(debug_dir, exist_ok=True)

    #     embedded_linear = dbg['embedded_linear'][0].detach().cpu().numpy()
    #     pos_enc = dbg['positional_encoding'][0].detach().cpu().numpy()
    #     after_pe = embedded[0].detach().cpu().numpy()

    #     pd.DataFrame(embedded_linear).to_csv(
    #         os.path.join(debug_dir, f'{dataset_name}_{fingerprint_type}_{args.target_assay}_embedded_linear_{safe_smiles}.csv'), index=False)
    #     pd.DataFrame(pos_enc).to_csv(
    #         os.path.join(debug_dir, f'{dataset_name}_{fingerprint_type}_{args.target_assay}_positional_encoding_{safe_smiles}.csv'), index=False)
    #     pd.DataFrame(after_pe).to_csv(
    #         os.path.join(debug_dir, f'{dataset_name}_{fingerprint_type}_{args.target_assay}_embedded_after_pe_{safe_smiles}.csv'), index=False)

    #     dbg_layers = dbg.get('debug_layers', [])
    #     pd.DataFrame(dbg_layers).to_csv(
    #         os.path.join(debug_dir, f'{dataset_name}_{fingerprint_type}_{args.target_assay}_block_stats_{safe_smiles}.csv'), index=False)

    #     for idx, A in enumerate(attn_all_layers):  # (1, H, L, L)
    #         A_np = A[0].detach().cpu().numpy()
    #         pd.DataFrame(A_np.mean(axis=0)).to_csv(
    #             os.path.join(debug_dir, f'{dataset_name}_{fingerprint_type}_{args.target_assay}_attn_layer{idx}_headavg_{safe_smiles}.csv'), index=False)
    #         pd.DataFrame(A_np[0]).to_csv(
    #             os.path.join(debug_dir, f'{dataset_name}_{fingerprint_type}_{args.target_assay}_attn_layer{idx}_head0_{safe_smiles}.csv'), index=False)

    #     print(f"Transformer validation output saved to: {debug_dir}")


def train_kfold_classification_model(train_loader, val_loader, model, args, device, fold=None,
                                     dataset_name=None, fingerprint_name=None, model_name=None,
                                     plot_curves=True):
    """
    classification model training
    training phase does not use early stopping, but monitors validation loss for early stopping

    Args:
        plot_curves: whether to plot learning curves, default True
        dataset_name: dataset name (used for learning curve title)
        fingerprint_name: fingerprint name (used for learning curve title)
        model_name: model name (used for learning curve title)

    Returns:
        trained_model
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay)

    # set warm-up parameters
    use_warmup = getattr(args, 'use_warmup', False)
    warmup_epochs = 50
    warmup_start_lr = args.lr * 1e-3  # original learning rate * 1e-3

    # only initialize scheduler if warm-up is not used
    # when using warm-up, scheduler will be initialized after warm-up ends
    if not use_warmup:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-8, last_epoch=-1
        )
    else:
        scheduler = None
        print(
            f"Enable Learning Rate Warm-up: first {warmup_epochs} epochs from {warmup_start_lr:.2e} linearly increase to {args.lr:.2e}")

    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_state = None  # save best weights

    epoch_pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss_avg:.4f}',
            'val_loss': f'{val_loss_avg:.4f}'
        })

        if use_warmup:
            if epoch < warmup_epochs:
                lr = warmup_start_lr + \
                    (args.lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif epoch == warmup_epochs:
                # warm-up ends, initialize CosineAnnealingLR (start from epoch 0)
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-8, last_epoch=-1
                )
                print(
                    f"Warm-up completed. Using CosineAnnealingLR from epoch {epoch+1}")
            else:
                scheduler.step()
        else:
            scheduler.step()

        if use_warmup and epoch < warmup_epochs:
            # warm-up phase: only update best_val_loss, do not perform early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        else:
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                # copy weights to CPU to avoid occupying GPU memory
                # save best weights
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                early_stop_counter = 0

            else:
                early_stop_counter += 1

            if early_stop_counter >= args.patience:
                print(
                    f"Early stopping based on validation loss at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}, LR: {current_lr:.2e}')

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_pred_prob = torch.sigmoid(y_pred)

            all_preds.extend(y_pred_prob.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    pred_binary = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, pred_binary)
    precision = precision_score(all_labels, pred_binary)
    recall = recall_score(all_labels, pred_binary)
    accuracy = accuracy_score(all_labels, pred_binary)
    auc_score = roc_auc_score(all_labels, all_preds)

    if fold is not None:
        print(f'Fold {fold+1} Validation: F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}')

    use_single_fold = False

    # plot learning curves
    if plot_curves and dataset_name and fingerprint_name and model_name and fold is not None:
        plot_learning_curves(train_losses, val_losses, fold, dataset_name,
                             fingerprint_name, model_name, use_single_fold, task_type='classification')

    return model


def train_kfold_regression_model(train_loader, val_loader, model, args, device, fold=None,
                                 dataset_name=None, fingerprint_name=None, model_name=None,):
    """
    regression model training
    training phase does not use early stopping, but monitors validation loss for early stopping

    Args:
        dataset_name: dataset name (used for learning curve title)
        fingerprint_name: fingerprint name (used for learning curve title)
        model_name: model name (used for learning curve title)

    Returns:
        tuple: (mse, rmse, mae, r2, all_labels, all_preds, trained_model)
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0

    inverse_transform = getattr(
        train_loader.dataset, 'inverse_transform_labels', None)

    epoch_pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss_avg:.4f}',
            'val_loss': f'{val_loss_avg:.4f}'
        })

        scheduler.step(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            early_stop_counter = 0
            # copy weights to CPU to avoid occupying GPU memory
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            print(
                f"Early stopping based on validation loss at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}, LR: {current_lr:.2e}')

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    if inverse_transform is not None:
        all_preds = inverse_transform(all_preds).flatten()
        all_labels = inverse_transform(all_labels).flatten()

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    if fold is not None:
        print(
            f'Fold {fold+1} Validation: MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

    use_single_fold = False

    plot_learning_curves(train_losses, val_losses, fold, dataset_name,
                         fingerprint_name, model_name, use_single_fold, task_type='regression')

    return model


def train_single_fold_classification_model(train_loader, val_loader, model, args, device, fold=None,
                                           dataset_name=None, fingerprint_name=None, model_name=None,):
    """
    classification model training (single fold version)
    training phase does not use early stopping, but monitors validation loss for early stopping

    Args:
        dataset_name: dataset name (used for learning curve title)
        fingerprint_name: fingerprint name (used for learning curve title)
        model_name: model name (used for learning curve title)

    Returns:
        trained_model
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7
    # )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-8, last_epoch=-1
    )
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_state = None

    epoch_pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss_avg:.4f}',
            'val_loss': f'{val_loss_avg:.4f}'
        })

        scheduler.step()

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            early_stop_counter = 0
            # copy weights to CPU to avoid occupying GPU memory
            # save best weights
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            print(
                f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            print(
                f"Early stopping based on validation loss at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}, LR: {current_lr:.2e}')

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_pred_prob = torch.sigmoid(y_pred)

            all_preds.extend(y_pred_prob.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_preds_binary = (all_preds > 0.5).astype(int)

    f1 = f1_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary)
    recall = recall_score(all_labels, all_preds_binary)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    auc_score = roc_auc_score(all_labels, all_preds)

    print(
        f'Validation: F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}')

    use_single_fold = True

    plot_learning_curves(train_losses, val_losses, fold,
                         dataset_name, fingerprint_name, model_name, use_single_fold, task_type='classification')

    return model


def train_single_fold_regression_model(train_loader, val_loader, model, args, device, fold=None,
                                       dataset_name=None, fingerprint_name=None, model_name=None,):
    """
    regression model training (single fold version)
    training phase does not use early stopping, but monitors validation loss for early stopping

    Args:
        dataset_name: dataset name (used for learning curve title)
        fingerprint_name: fingerprint name (used for learning curve title)
        model_name: model name (used for learning curve title)

    Returns:
        tuple: (mse, rmse, mae, r2, all_labels, all_preds, trained_model)
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_state = None

    inverse_transform = getattr(
        train_loader.dataset, 'inverse_transform_labels', None)

    epoch_pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss_avg:.4f}',
            'val_loss': f'{val_loss_avg:.4f}'
        })

        scheduler.step(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            early_stop_counter = 0
            # copy weights to CPU to avoid occupying GPU memory
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            print(
                f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}")

        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            print(
                f"Early stopping based on validation loss at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}, LR: {current_lr:.2e}')

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    if inverse_transform is not None:
        all_preds = inverse_transform(all_preds).flatten()
        all_labels = inverse_transform(all_labels).flatten()

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(
        f'Validation: MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

    use_single_fold = True

    plot_learning_curves(train_losses, val_losses, fold,
                         dataset_name, fingerprint_name, model_name, use_single_fold, task_type='regression')

    return model
