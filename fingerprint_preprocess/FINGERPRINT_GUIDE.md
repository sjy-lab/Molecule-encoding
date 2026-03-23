# Fingerprint Precompute — Usage Guide

This directory contains three scripts that convert molecular SMILES into fingerprint vectors stored as `.npz` files, and validate the correctness of the generated outputs.

---

## Script Overview

| Script | Function | Output |
|---|---|---|
| `precompute_PubChem.py` | Queries the PubChem API to retrieve 881-bit CACTVS fingerprints | `{dataset}_PubChem_embeddings.npz` |
| `precompute_SMARTS.py` | Generates SMARTS strings via RDKit, then applies TF-IDF vectorization | `{dataset}_SMARTS_embeddings.npz` |
| `validate_fingerprint_npz.py` | Validates that `.npz` SMILES match the source CSV | Terminal report (exit 0 = all passed) |

---

## 1. `precompute_PubChem.py`

Uses `pubchempy` to call the PubChem API and retrieve the 881-bit CACTVS fingerprint for each SMILES.
Molecules for which no fingerprint can be retrieved are filled with a zero vector, and the count is recorded in the `.npz` metadata.

**Note**: Each SMILES requires one network API call, so processing large datasets can take a long time.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `tox21` | Dataset name (used in the output filename) |
| `--dataset_path` | *(absolute path)* | Path to the input CSV; must contain a `standardised_smiles` column |
| `--output_dir` | *(absolute path)* | Output directory |

### Example

```bash
python precompute_PubChem.py \
  --dataset_name BBBP \
  --dataset_path ../../src/data/Processed_dataset/BBBP_aggregated.csv \
  --output_dir ./PubChem
```

---

## 2. `precompute_SMARTS.py`

Converts SMILES to SMARTS strings using RDKit, then vectorizes them with `CountVectorizer + TfidfTransformer`.
No external API calls are required, making this significantly faster than the PubChem approach.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `Tox21` | Dataset name |
| `--dataset_path` | *(absolute path)* | Path to the input CSV; must contain a `standardised_smiles` column |
| `--output_dir` | *(absolute path)* | Output directory |

### Example

```bash
python precompute_SMARTS.py \
  --dataset_name BBBP \
  --dataset_path ../../src/data/Processed_dataset/BBBP_aggregated.csv \
  --output_dir ./SMARTS
```
---

## 3. `validate_fingerprint_npz.py`

Validates that a generated `.npz` file is fully aligned with the source CSV. Checks performed:

1. File exists
2. `.npz` contains both `fingerprints` and `smiles` keys
3. Row count matches the CSV
4. Each SMILES string matches exactly
5. Fingerprint matrix row count matches the CSV

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv_dir` | `src/data/Processed_dataset` | Directory containing the source CSV files |
| `--fp_dir` | `fingerprint_preprocess` | Directory containing `.npz` files (with subdirectories `PubChem/`, `SMARTS/`, etc.) |
| `--datasets` | all 7 datasets | Datasets to validate, space-separated |
| `--fingerprints` | `PubChem SMARTS InChI` | Fingerprint types to validate, space-separated |
| `--smiles_column` | `standardised_smiles` | Name of the SMILES column in the CSV |

### Example

```bash
# Validate specific datasets and fingerprint types
python fingerprint_preprocess/validate_fingerprint_npz.py \
  --datasets BBBP BACE \
  --fingerprints PubChem SMARTS
```