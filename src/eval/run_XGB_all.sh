

cd /home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/src/eval/base_model/

RESULTS_PATH="experiments/results_XGB_sklearn"

echo "start to evaluate XGBoost (sklearn API)..."
echo "results will be saved to: $RESULTS_PATH"

echo "processing BBBP dataset..."
DATA_PATH="../../data/Processed_dataset/BBBP_aggregated.csv"
TARGET_ASSAY="p_np"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset BBBP \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4


python -u evaluate_XGB_sklearn.py \
    --dataset BBBP \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset BBBP \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset BBBP \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset BBBP \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset BBBP \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4


DATA_PATH="../../data/Processed_dataset/ToxCast_aggregated.csv"
TARGET_ASSAY="NVS_ADME_hCYP2C19"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset ToxCast \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ToxCast \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ToxCast \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ToxCast \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ToxCast \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ToxCast \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

DATA_PATH="../../data/Processed_dataset/SIDER_aggregated.csv"
TARGET_ASSAY="Nervous system disorders"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset SIDER \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset SIDER \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset SIDER \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset SIDER \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset SIDER \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset SIDER \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

DATA_PATH="../../data/Processed_dataset/ClinTox_aggregated.csv"
TARGET_ASSAY="CT_TOX"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset ClinTox \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ClinTox \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ClinTox \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ClinTox \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ClinTox \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset ClinTox \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --use_feature_net


DATA_PATH="../../data/Processed_dataset/ESOL_aggregated.csv"
TARGET_ASSAY="measured log solubility in mols per litre"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset ESOL \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset ESOL \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset ESOL \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset ESOL \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset ESOL \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4

python -u evaluate_XGB_sklearn.py \
    --dataset ESOL \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --requires_imputation \
    --n_jobs 4


DATA_PATH="../../data/Processed_dataset/MUTAG_aggregated.csv"
TARGET_ASSAY="label"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset MUTAG \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH"

python -u evaluate_XGB_sklearn.py \
    --dataset MUTAG \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH"


python -u evaluate_XGB_sklearn.py \
    --dataset MUTAG \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH"


python -u evaluate_XGB_sklearn.py \
    --dataset MUTAG \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH"

python -u evaluate_XGB_sklearn.py \
    --dataset MUTAG \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH"

python -u evaluate_XGB_sklearn.py \
    --dataset MUTAG \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH"


DATA_PATH="../../data/Processed_dataset/BACE_aggregated.csv"
TARGET_ASSAY="Class"
SMILES_COL="standardised_smiles"

python -u evaluate_XGB_sklearn.py \
    --dataset BACE \
    --fingerprint MACCS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset BACE \
    --fingerprint ECFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset BACE \
    --fingerprint FCFP \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset BACE \
    --fingerprint RDKit \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset BACE \
    --fingerprint PubChem \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --use_feature_net

python -u evaluate_XGB_sklearn.py \
    --dataset BACE \
    --fingerprint SMARTS \
    --multi_assay_data "$DATA_PATH" \
    --target_assay "$TARGET_ASSAY" \
    --smiles_column "$SMILES_COL" \
    --results_path "$RESULTS_PATH" \
    --use_feature_net

echo "results file is located at: $RESULTS_PATH"
