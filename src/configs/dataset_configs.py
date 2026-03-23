from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    hidden_dims: List[int]  # hidden layer dim of MLP and Transformer+MLP
    dropout_rates: List[float]
    activation: str
    use_normalization: bool
    num_heads: Optional[int] = None
    transformer_dim: Optional[int] = None  # feedforward dim of Transformer
    transformer_dropout: float = 0.1  # dropout rate of Transformer
    num_layers: Optional[int] = None  # Transformer encoder layer


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    patience: int = 5
    n_splits: int = 5


@dataclass
class DatasetConfig:
    task_type: str  # 'classification' or 'regression'
    input_dim: Optional[int]
    output_dim: int
    model_config: ModelConfig
    training_config: TrainingConfig
    use_feature_net: bool = True  # whether use featurenet or not
    # read the data from imputation or only aggregated data
    requires_imputation: bool = True


@dataclass
class FingerprintConfig:
    preprocessing: Optional[dict] = None


FINGERPRINT_CONFIGS = {
    'MACCS': FingerprintConfig(),
    'ECFP': FingerprintConfig(),
    'FCFP': FingerprintConfig(),
    'RDKit': FingerprintConfig(),
    'PubChem': FingerprintConfig(
        preprocessing={
            'type': 'precomputed',
            'file_pattern': 'fingerprint_preprocess/PubChem/{dataset}_PubChem_embeddings.npz'
        }
    ),
    'SMARTS': FingerprintConfig(
        preprocessing={
            'type': 'precomputed',
            'file_pattern': 'fingerprint_preprocess/SMARTS/{dataset}_SMARTS_embeddings.npz'
        }
    ),
    'InChI': FingerprintConfig(
        preprocessing={
            'type': 'precomputed',
            'file_pattern': 'fingerprint_preprocess/InChI/{dataset}_InChI_embeddings.npz'
        }
    ),
}

DATASET_CONFIGS = {

    'BBBP': DatasetConfig(
        task_type='classification',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            dropout_rates=[0.4, 0.4, 0.4],
            activation='LeakyReLU',
            use_normalization=True,
            hidden_dims=[512, 256, 64],
            num_heads=1,
            transformer_dim=512,
            transformer_dropout=0.1,
            num_layers=5,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=0.00001,
            weight_decay=0.001,
            patience=10
        ),
        use_feature_net=False
    ),
    'MUTAG': DatasetConfig(
        task_type='classification',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            dropout_rates=[0.3, 0.3, 0.3],
            activation='LeakyReLU',
            use_normalization=True,
            hidden_dims=[256, 128, 32],
            num_heads=2,
            transformer_dim=256,
            transformer_dropout=0.1,
            num_layers=5,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=16,
            learning_rate=0.0001,
            weight_decay=0.001 
        ),
        use_feature_net=False,
        requires_imputation=False
    ),
    'ClinTox': DatasetConfig(
        task_type='classification',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            dropout_rates=[0.3, 0.2],
            activation='LeakyReLU',
            use_normalization=True,
            hidden_dims=[256, 128],
            num_heads=8,
            transformer_dim=256,
            transformer_dropout=0.1,
            num_layers=1,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.01,
            patience=10
        ),
        use_feature_net=True,
        requires_imputation=True
    ),
    'ToxCast': DatasetConfig(
        task_type='classification',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            hidden_dims=[2048, 1024, 512, 256], 
            dropout_rates=[0.5, 0.4, 0.3, 0.2], 
            activation='LeakyReLU',
            use_normalization=False,
            transformer_dropout=0.1,
            num_heads=1,
            transformer_dim=256,
            num_layers=2,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=0.0001,
            weight_decay=0.01,
            patience=10
        )
    ),
    'SIDER': DatasetConfig(
        task_type='classification',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            hidden_dims=[256, 128, 32], 
            dropout_rates=[0.6, 0.6, 0.6],  
            activation='LeakyReLU',
            use_normalization=False,
            num_heads=2,
            transformer_dim=256, 
            transformer_dropout=0.1,
            num_layers=3,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=16,
            learning_rate=0.0005,
            weight_decay=0.005, 
            patience=15
        )
    ),
    'BACE': DatasetConfig(
        task_type='classification',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            dropout_rates=[0.3, 0.3, 0.3],     
            activation='LeakyReLU',               
            use_normalization=True,
            hidden_dims=[512, 256, 16],          
            num_heads=1,                          
            transformer_dim=512,                  
            transformer_dropout=0.1,
            num_layers=3,                         
        ),
        training_config=TrainingConfig(
            epochs=100,                          
            batch_size=64,                      
            learning_rate=1e-05,                
            weight_decay=1e-05,                   
            patience=15                          
        ),
        use_feature_net=False
    ),

    'ESOL': DatasetConfig(
        task_type='regression',
        input_dim=None,
        output_dim=1,
        model_config=ModelConfig(
            dropout_rates=[0.1, 0.1, 0.1],
            activation='LeakyReLU',
            use_normalization=True,
            hidden_dims=[128, 64, 32],
            num_heads=1,
            transformer_dim=128,
            transformer_dropout=0.1,
            num_layers=2,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=1e-05,
            weight_decay=0.0001
        ),
        use_feature_net=False,
        requires_imputation=True
    )
}

DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_model_config(dataset_name: str, fingerprint_type: str, model_type: str) -> tuple:
    """
    Get model config setting

    Args:
        dataset_name
        fingerprint_type
        model_type

    Returns:
        tuple: (dataset_config, fingerprint_config)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    if fingerprint_type not in FINGERPRINT_CONFIGS:
        raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    fingerprint_config = FINGERPRINT_CONFIGS[fingerprint_type]

    return dataset_config, fingerprint_config


FINGERPRINT_TYPES = list(FINGERPRINT_CONFIGS.keys())
