import os
import torch
import numpy as np
from torch.utils.data import Dataset
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from src.configs.dataset_configs import FINGERPRINT_CONFIGS


class BaseMolecularDataset(Dataset):

    def __init__(self, fingerprint_type='MACCS', mode='train'):
        super(BaseMolecularDataset, self).__init__()
        self.mode = mode
        self.fingerprint_type = fingerprint_type
        self._scaler = None
        self.data = None
        self.labels = None
        self.dim = None

    def extract_features(self, mol_list):
        """According to the fingerprint type, extract features"""
        if self.fingerprint_type == 'MACCS':
            return [list(MACCSkeys.GenMACCSKeys(mol)) for mol in mol_list]
        elif self.fingerprint_type == 'ECFP':
            return [list(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)) for mol in mol_list]
        elif self.fingerprint_type == 'FCFP':
            return [list(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useFeatures=True)) for mol in mol_list]
        elif self.fingerprint_type in ['PubChem', 'SMARTS', 'InChI']:
            return self.load_precomputed_fingerprints(mol_list)
        elif self.fingerprint_type == 'RDKit':
            return [list(RDKFingerprint(mol)) for mol in mol_list]
        else:
            raise ValueError(
                f"Unsupported fingerprint type: {self.fingerprint_type}")

    def load_precomputed_fingerprints(self, mol_list):
        """Load precomputed fingerprints (PubChem、SMARTS)"""
        if not self.dataset_name:
            raise ValueError(
                f"When using {self.fingerprint_type} fingerprints, dataset_name must be provided")

        if not self.fingerprint_config or not self.fingerprint_config.preprocessing:
            raise ValueError(
                f"{self.fingerprint_type} fingerprints are not configured for precomputed mode")

        file_pattern = self.fingerprint_config.preprocessing['file_pattern']
        embedding_file = file_pattern.format(dataset=self.dataset_name)

        if not os.path.isabs(embedding_file):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            embedding_file = os.path.join(project_root, embedding_file)

        if not os.path.exists(embedding_file):
            raise FileNotFoundError(
                f"File not found: {self.fingerprint_type} embedding: {embedding_file}")

        print(f"Loading {self.fingerprint_type} embeddings: {embedding_file}")
        data = np.load(embedding_file, allow_pickle=True)
        embedding_weights = data['fingerprints']
        print(
            f"{self.fingerprint_type} fingerprint dimension: {embedding_weights.shape}")

        return embedding_weights

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    @property
    def scaler(self):
        """Get scaler"""
        return self._scaler

    def inverse_transform_labels(self, y):
        """Convert standardized labels back to original scale (only applicable for regression tasks)"""
        if self._scaler is None:
            return y

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        return self._scaler.inverse_transform(y.reshape(-1, 1))
