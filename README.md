# FeatureNet Molecular Embedding

This is the code repository for the paper **"A systematic investigation of molecular encoding methods for drug property predictions across neural network and Transformer encoder-based model"** by Sheng-Ya Chen and Shan-Ju Yeh.

---

## Overview

This repository implements a molecular property prediction framework that combines molecular fingerprints with multi-assay bioactivity data. It supports two model architectures:

- **MLP** — Multi-layer Perceptron
- **MLP+TL** — MLP with a Transformer encoder layer (attention-based)

### Supported Datasets

| Dataset  | Task           | Feature Network |
|----------|----------------|-----------------|
| BBBP     | Classification | No              |
| BACE     | Classification | Yes             |
| ClinTox  | Classification | Yes             |
| SIDER    | Classification | Yes             |
| MUTAG    | Classification | No              |
| ToxCast  | Classification | Yes             |
| B3DB     | Classification | No              |
| ESOL     | Regression     | No              |

### Supported Fingerprints

| Fingerprint | Type          | Precompute Required |
|-------------|---------------|---------------------|
| MACCS       | Substructure-based    | No                  |
| ECFP        | Circular      | No                  |
| FCFP        | Feature-based | No                  |
| RDKit       | Topological   | No                  |
| PubChem     | CACTVS 881-bit | Yes (API)          |
| SMARTS      | TF-IDF        | Yes                 |

---

## Installation

```bash
conda env create -f environment.yml
conda activate tuna
```

---

## Project Structure

```
FeatureNet_mol_embedding_ya/
├── main.py                         # Main training script
├── evaluate_test_predictions.py    # Test set evaluation (k-fold)
├── environment.yml                 # Conda environment
├── src/
│   ├── configs/
│   │   ├── dataset_configs.py      # Dataset & model hyperparameters
│   │   └── config.md               # Final experiment parameter records
│   ├── data/
│   │   ├── standardisation_aggregation.py   # SMILES standardisation
│   │   ├── impute_missing_values.py         # Multi-assay imputation
│   │   └── Processed_dataset/              # Cleaned CSVs
│   ├── datasets/
│   │   └── base_dataset.py         # Fingerprint extraction base class
│   ├── models/
│   │   └── neural_networks.py      # NeuralNet & TransformerMLPModel
│   ├── utils/
│   │   └── training_utils_improved.py  # Training loops, evaluation, plotting
│   └── eval/
│       ├── evaluate_xgb.py         # XGBoost baseline
│       └── base_model/             # RF / XGBoost sklearn baselines
├── fingerprint_preprocess/
│   ├── precompute_PubChem.py       # PubChem API fingerprint precompute
│   ├── precompute_SMARTS.py        # SMARTS TF-IDF precompute
│   ├── validate_fingerprint_npz.py # Validate .npz fingerprint files
│   └── README.md                   # Fingerprint precompute guide
├── toxcast-curation/
│   └── toxcast_binary_from_ctx_api.py  # ToxCast dataset curation via API
└── experiments/                    # Auto-generated experiment outputs
```

---

## Usage

### Step 1 — Precompute Fingerprints (PubChem / SMARTS only)

For fingerprints that require precomputation, run the corresponding script before training.
See [`fingerprint_preprocess/FINGERPRINT_GUIDE.md`](fingerprint_preprocess/FINGERPRINT_GUIDE.md) for details.

```bash
python fingerprint_preprocess/precompute_PubChem.py \
  --dataset_name BBBP \
  --dataset_path src/data/Processed_dataset/BBBP_aggregated.csv \
  --output_dir fingerprint_preprocess/PubChem

python fingerprint_preprocess/precompute_SMARTS.py \
  --dataset_name BBBP \
  --dataset_path src/data/Processed_dataset/BBBP_aggregated.csv \
  --output_dir fingerprint_preprocess/SMARTS
```

### Step 2 — Configure Hyperparameters

Model and training hyperparameters are defined in `src/configs/dataset_configs.py`.
Refer to `src/configs/config.md` for the final experiment configurations used in the paper.
Copy the relevant dataset/model/fingerprint settings into `dataset_configs.py` before running.

### Step 3 — Train (K-Fold Cross-Validation)

```bash
python main.py \
  --dataset BBBP \
  --multi_assay_data src/data/Processed_dataset/BBBP_aggregated.csv \
  --target_assay p_np \
  --model_type MLP \
  --fingerprint MACCS
```

**Output** (under `experiments/{dataset}/{model}/kfold/`):
- Per-fold prediction CSV (`*_fold{n}_test.csv`)
- Learning curve plots
- Experiment configuration JSON

### Step 4 — Evaluate Test Set

```bash
python evaluate_test_predictions.py --dataset BBBP --model MLP
```

**Output** (under `experiments/{dataset}/{model}/kfold/result_test/`):
- ROC curve plot
- Per-fold detailed metrics CSV
- Fold-averaged metrics CSV
- Consolidated comparison table

---

## Additional Modes

### Single-Molecule Prediction + Attention Heatmap (MLP+TL)

Trains on all data except the specified molecule, then outputs the Transformer attention heatmap for that molecule.

```bash
python main.py \
  --dataset BBBP \
  --multi_assay_data src/data/Processed_dataset/BBBP_aggregated.csv \
  --target_assay p_np \
  --model_type MLP+TL \
  --fingerprint MACCS \
  --use_single_fold \
  --test_id "<SMILES string>"
```

**Output** (under `experiments/{dataset}/MLP+TL/`):
- Attention heatmap PNG
- Attention weight matrix CSV
- Embedded vector CSV

### Save Model Weights

```bash
python main.py \
  --dataset BBBP \
  --multi_assay_data src/data/Processed_dataset/BBBP_aggregated.csv \
  --target_assay p_np \
  --model_type MLP \
  --fingerprint MACCS \
  --use_single_fold
```

**Output**: model weights saved to `experiments/{dataset}/{model}/models/`
