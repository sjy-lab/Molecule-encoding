
- ToxCast
    - FCFP
        - MLP
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256, 128], 
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
            ```
            
        - MLP+TL
            
            ```python
            'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256, 128], 
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
                        batch_size=32,
                        learning_rate=1e-06,
                        weight_decay=1e-05,
                        patience=10
                    )
                ),
            ```
            
    - ECFP
        - MLP
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 256, 128, 64], 
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
            ```
            
        - MLP+TL
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 256, 128, 64], 
                        dropout_rates=[0.5, 0.4, 0.3, 0.2], 
                        activation='LeakyReLU',
                        use_normalization=False,
                        transformer_dropout=0.1,
                        num_heads=1,
                        transformer_dim=512,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-06,
                        weight_decay=1e-05,
                        patience=10
                    )
                ),
            
            ```
            
    - RDKit
        - MLP
            
            ```python
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
            ```
            
        - MLP+TL
            
            ```python
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
                        transformer_dim=1024,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-06,
                        weight_decay=1e-02,
                        patience=10
                    )
                ),
            
            ```
            
    - PubChem
        - MLP
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 256, 128, 64], 
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
            ```
            
        - MLP+TL
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 256, 128, 64], 
                        dropout_rates=[0.5, 0.4, 0.3, 0.2], 
                        activation='LeakyReLU',
                        use_normalization=False,
                        transformer_dropout=0.1,
                        num_heads=1,
                        transformer_dim=512,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=0.000001,
                        weight_decay=0.00001,
                        patience=10
                    )
                ),
            ```
            
    - MACCS
        - MLP
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256, 128], 
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
            ```
            
        - MLP+TL
            
            ```python
                'ToxCast': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256, 128], 
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
                        learning_rate=0.000001,
                        weight_decay=0.00001,
                        patience=10
                    )
                ),
            ```
            
    - SMARTS
        - MLP
            
            ```python
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
            ```
            
        - MLP+TL
            
            ```python
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
                        transformer_dim=1024,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.0001,
                        patience=10
                    )
                ),
            ```
            
- ClinTox
    - FCFP
        - MLP
            
            ```python
                'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.2],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512],
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
            ```
            
        - MLP+TL
            
            ```python
            'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.2],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=3,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.01,
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - ECFP
        - MLP
            
            ```python
                'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.2],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512],
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
            ```
            
        - MLP+TL
            
            ```python
            'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.2],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.01,
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - RDKit
        - MLP
            
            ```python
                'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.5, 0.5, 0.5],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[2048, 1024, 512],
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
            ```
            
        - MLP+TL
            
            ```python
            'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.5, 0.5, 0.5],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[2048, 1024, 512],
                        num_heads=1,
                        transformer_dim=2048,
                        transformer_dropout=0.1,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-06,
                        weight_decay=0.01, 
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - PubChem
        - MLP
            
            ```python
                'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512],
                        num_heads=1,
                        transformer_dim=2048,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=4
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-04,
                        weight_decay=0.01,
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
        - MLP+TL
            
            ```python
            'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.001,
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - MACCS
        - MLP
            
            ```python
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
            ```
            
        - MLP+TL
            
            ```python
            'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 128],
                        num_heads=1,
                        transformer_dim=256,
                        transformer_dropout=0.1,
                        num_layers=6,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.001,
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - SMARTS
        - MLP
            
            ```python
                'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[128, 64],
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
            ```
            
        - MLP+TL
            
            ```python
            'ClinTox': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[128, 64],
                        num_heads=1,
                        transformer_dim=256,
                        transformer_dropout=0.1,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-05,
                        weight_decay=0.01,
                        patience=10
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    
- MUTAG
    - FCFP
        - MLP
            
            ```python
                'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 128],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-04,
                        weight_decay=0.01
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            DatasetConfig：
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 128],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-05,
                        weight_decay=0.01 
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
    - ECFP
        - MLP
            
            ```python
                'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 128],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-04,
                        weight_decay=0.01 
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 128],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.0001
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
    - RDKit
        - MLP
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[2048, 1024, 512],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-04,
                        weight_decay=0.001
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[2048, 1024, 512],
                        num_heads=1,
                        transformer_dim=2048,
                        transformer_dropout=0.1,
                        num_layers=2,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-05,
                        weight_decay=0.001 
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
    - PubChem
        - MLP
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-04,
                        weight_decay=0.01
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.01
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
    - MACCS
        - MLP
            
            ```python
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
            ```
            
        - MLP+TL
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 128, 32],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.001
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
    - SMARTS
        - MLP
            
            ```python
                'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 128, 64],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=5,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=1e-04,
                        weight_decay=0.01
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'MUTAG': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 128, 64],
                        num_heads=1,
                        transformer_dim=256,
                        transformer_dropout=0.1,
                        num_layers=5,
                    
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.01
                    ),
                    use_feature_net=False,
                    requires_imputation=False
                ),
            ```
            
- BBBP
    - MACCS
        - MLP
            
            ```python
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
                        num_layers=2,
            
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.00001,
                        weight_decay=0.001,
                        patience=10
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
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
            ```
            
    - ECFP
        - MLP
            
            ```python
                'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4], 
                        #[0.5, 0.4, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 16], 
                        #F1: 0.917 ± 0.013
                        #AUC: 0.902 ± 0.026
                        num_heads=4,             
                        transformer_dim=512,      
                        transformer_dropout=0.1,
                        num_layers=4,               
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,          
                        learning_rate=0.0001, 
                        weight_decay=0.01,   
                        patience= 15     
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 16],          
                        num_heads=1,                        
                        transformer_dim=256,              
                        transformer_dropout=0.1,
                        num_layers=5,                                        
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=64,                       
                        learning_rate=0.000005,                
                        weight_decay=0.001,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - PubChem
        - MLP
            
            ```python
             'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 128, 16],          
                        num_heads=1,                        
                        transformer_dim=256,                 
                        transformer_dropout=0.1,
                        num_layers=2,                                            
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=16,                       
                        learning_rate=0.00005,                
                        weight_decay=0.001,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                )
            ```
            
        - MLP+TL
            
            ```python
            'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 128, 16],          
                        num_heads=1,                        
                        transformer_dim=256,                 
                        transformer_dropout=0.1,
                        num_layers=2,                                        
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=64,                       
                        learning_rate=0.000005,                
                        weight_decay=0.00001,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - FCFP
        - MLP
            
            ```python
            'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None, 
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 16],          
                        num_heads=1,                        
                        transformer_dim=256,              
                        transformer_dropout=0.1,
                        num_layers=5,                                        
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                       
                        learning_rate=0.00005,                
                        weight_decay=0.001,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None, 
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 16],          
                        num_heads=1,                        
                        transformer_dim=256,              
                        transformer_dropout=0.1,
                        num_layers=5,                                        
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                       
                        learning_rate=0.00005,                
                        weight_decay=0.001,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - SMARTS
        - MLP
            
            ```python
            'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 64, 16],          
                        num_heads=1,                         
                        transformer_dim=512,             
                        transformer_dropout=0.1,
                        num_layers=2,                       
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                         
                        batch_size=16,                       
                        learning_rate=0.0001,               
                        weight_decay=0.005,                  
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
                'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None, 
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 64, 16],          
                        num_heads=1,                         
                        transformer_dim=512,             
                        transformer_dropout=0.1,
                        num_layers=4,                       
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
            ```
            
    - RDKit
        - MLP
            
            ```python
                'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 16],          
                        num_heads=1,                        
                        transformer_dim=256,              
                        transformer_dropout=0.1,
                        num_layers=2,                                           
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                       
                        learning_rate=0.0001,                
                        weight_decay=0.005,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            
                'BBBP': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.4, 0.4, 0.4],      
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
                        batch_size=32,                       
                        learning_rate=0.000001,                
                        weight_decay=0.005,                 
                        patience=10                          
                    ),
                    use_feature_net=False
                ),
            ```
            
- SIDER
    - FCFP
        - MLP
            
            ```python
            'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256],    
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',
                       use_normalization=True,
                        num_heads=1,                      
                        transformer_dim=512,             
                        num_layers=2,                     
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                       
                        batch_size=32,                    
                        learning_rate=0.00001,             
                        weight_decay=0.001,              
                        patience=10                       
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
        - MLP+TL
            
            ```python
            'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256],    
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                      
                        transformer_dim=512,              
                        num_layers=2,                     
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                       
                        batch_size=64,                    
                        learning_rate=1e-05,             
                        weight_decay=0.01,              
                        patience=10                       
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - ECFP
        - MLP
            
            ```python
            'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256],  
                        dropout_rates=[0.3, 0.3, 0.3], 
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                      
                        transformer_dim=512,              
                        num_layers=2,                     
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                       
                        batch_size=32,                    
                        learning_rate=0.00001,             
                        weight_decay=0.001,              
                        patience=10                      
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
        - MLP+TL
            
            ```python
            'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256],  
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                     
                        transformer_dim=1024,              
                        num_layers=4,                    
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                      
                        batch_size=64,                   
                        learning_rate=1e-05,            
                        weight_decay=0.01,              
                        patience=10                     
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - RDKit
        - MLP
            
            ```python
                'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[2048, 1024, 512],
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,
                        transformer_dim=512,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.00001,
                        weight_decay=0.01,
                        patience=15
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
        - MLP+TL
            
            ```python
            'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[2048, 1024, 512],    
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                      
                        transformer_dim=1024,              
                        num_layers=4,                     
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                       
                        batch_size=16,                    
                        learning_rate=1e-05,             
                        weight_decay=0.01,              
                        patience=10                      
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - PubChem
        - MLP
            
            ```python
             'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 128, 64],   
                        dropout_rates=[0.3, 0.3, 0.3], 
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                   
                        transformer_dim=512,        
                        num_layers=2,                
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                 
                        batch_size=32,          
                        learning_rate=0.0001,     
                        weight_decay=0.001,    
                        patience=10            
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
        - MLP+TL
            
            ```python
                'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 128, 64],  
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                     
                        transformer_dim=512,            
                        num_layers=2,                   
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                      
                        batch_size=32,                    
                        learning_rate=0.00001,             
                        weight_decay=0.001,          
                        patience=10                      
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
    - MACCS
        - MLP
            
            ```python
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
            ```
            
        - MLP+TL
            
            ```python
                'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[256, 128, 32],
                        dropout_rates=[0.6, 0.6, 0.6],
                        activation='LeakyReLU',
                        use_normalization=False,
                        num_heads=1,
                        ansformer_dim=256,
                        transformer_dropout=0.1,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=0.00005,
                        weight_decay=0.001,
                        patience=15
                    )
                ),
            ```
            
        
    - SMARTS
        - MLP
            
            ```python
                'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[512, 256, 64],
                        dropout_rates=[0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,
                        transformer_dim=512,
                        num_layers=2,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.00001,
                        weight_decay=0.01,
                        patience=15
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            ```
            
        - MLP+TL
            
            ```python
            'SIDER': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        hidden_dims=[1024, 512, 256],   
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',
                        use_normalization=True,
                        num_heads=1,                      
                        transformer_dim=1024,              
                        num_layers=6,                     
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                       
                        batch_size=64,                    
                        learning_rate=1e-05,             
                        weight_decay=0.01,              
                        patience=10                       
                    ),
                    use_feature_net=True,
                    requires_imputation=True
                ),
            
            ```
            
- BACE
    - FCFP
        - MLP
            
            ```python
            
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[512, 128, 32],          
                        num_heads=1,                          
                        transformer_dim=512,                  
                        transformer_dropout=0.1,
                        num_layers=2,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                      
                        learning_rate=0.0001,                
                        weight_decay=0.01,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            
            ```
            
        - MLP+TL
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[512, 128, 32],          
                        num_heads=1,                          
                        transformer_dim=512,                  
                        transformer_dropout=0.1,
                        num_layers=3,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                      
                        learning_rate=0.000001,                
                        weight_decay=0.0001,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - ECFP
        - MLP
            
            ```python
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
                        num_layers=2,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                      
                        learning_rate=0.0001,                
                        weight_decay=0.001,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
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
            ```
            
    - RDKit
        - MLP
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3 , 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[1024, 256, 64],          
                        num_heads=1,                          
                        transformer_dim=512,                  
                        transformer_dropout=0.1,
                        num_layers=2,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                      
                        learning_rate=0.0001,                
                        weight_decay=0.01,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[1024, 256, 64],          
                        num_heads=1,                          
                        transformer_dim=1024,                  
                        transformer_dropout=0.1,
                        num_layers=2,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=16,                      
                        learning_rate=1e-06,                
                        weight_decay=1e-05,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - PubChem
        - MLP
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[1024, 256, 64],          
                        num_heads=1,                          
                        transformer_dim=512,                  
                        transformer_dropout=0.1,
                        num_layers=2,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                      
                        learning_rate=0.0001,                
                        weight_decay=0.01,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
        - MLP+TL
            
            ```python
            'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[512, 256, 64],          
                        num_heads=1,                          
                        transformer_dim=512,                  
                        transformer_dropout=0.1,
                        num_layers=4,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=64,                      
                        learning_rate=0.000001,                
                        weight_decay=0.001,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - MACCS
        - MLP
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],    
                        activation='LeakyReLU',             
                        use_normalization=True,
                        hidden_dims=[256, 128, 64],    
                        num_heads=1,                     
                        transformer_dim=512,        
                        transformer_dropout=0.1,
                        num_layers=3,                
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                        
                        batch_size=16,                   
                        learning_rate=0.0001,        
                        weight_decay=0.01,           
                        patience=15                  
                    ),
                ),
            ```
            
        - MLP+TL
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[256, 128, 64],          
                        num_heads=1,                          
                        transformer_dim=256,                  
                        transformer_dropout=0.1,
                        num_layers=3,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=16,                      
                        learning_rate=0.00001,                
                        weight_decay=0.0001,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                ),
            ```
            
    - SMARTS
        - MLP
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],  
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[1024, 128, 32], 
                        num_heads=1,                          
                        transformer_dim=512,                  
                        transformer_dropout=0.1,
                        num_layers=2,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=16,            
                        learning_rate=0.0001,                
                        weight_decay=0.01,                   
                        patience=15                          
                    ),
                    use_feature_net=False
                )
            ```
            
        - MLP+TL
            
            ```python
                'BACE': DatasetConfig(
                    task_type='classification',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates=[0.3, 0.3, 0.3],     
                        activation='LeakyReLU',               
                        use_normalization=True,
                        hidden_dims=[1024, 128, 32],          
                        num_heads=1,                          
                        transformer_dim=1024,                  
                        transformer_dropout=0.1,
                        num_layers=3,                         
                    ),
                    training_config=TrainingConfig(
                        epochs=100,                          
                        batch_size=32,                      
                        learning_rate=0.00001,                
                        weight_decay=0.00001,                   
                        patience=15                          
                    ),
                ),
            ```
            
    
- ESOL
    - FCFP
        - MLP
            
            ```python
                'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.3, 0.3, 0.3],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=2,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=4,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=0.0001,
                        weight_decay=0.0001,
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                )
            ```
            
        - MLP+TL
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-06,
                        weight_decay=1e-05 
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                ),
            ```
            
    - ECFP
        - MLP
            
            ```python
                'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=2,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=4,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=0.0001,
                        weight_decay=0.0001,
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                )
            ```
            
        - MLP+TL
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=5,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=1e-05
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                ),
            ```
            
    - RDKit
        - MLP
            
            ```python
                'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024,512,256],
                        num_heads=2,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.0001,
                        weight_decay=0.0001 
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                )
            ```
            
        - MLP+TL
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[1024, 512, 256],
                        num_heads=1,
                        transformer_dim=1024,
                        transformer_dropout=0.1,
                        num_layers=3,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-06,
                        weight_decay=1e-05
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                ),
            ```
            
    - PubChem
        - MLP
            
            ```python
                'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 128],
                        num_heads=2,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.0001,
                        weight_decay=0.0001,
                        patience=15
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                )
            ```
            
        - MLP+TL
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[512, 256, 128],
                        num_heads=1,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=3,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=16,
                        learning_rate=1e-06,
                        weight_decay=1e-05 
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                ),
            ```
            
    - MACCS
        - MLP
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 128, 64],
                        num_heads=2,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=4,
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.0001,
                        weight_decay=0.01
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                )
            ```
            
        - MLP+TL
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[256, 128, 64],
                        num_heads=1,
                        transformer_dim=256,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=1e-05 
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                ),
            ```
            
    - SMARTS
        - MLP
            
            ```python
                'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[128, 64, 32],
                        num_heads=2,
                        transformer_dim=512,
                        transformer_dropout=0.1,
                        num_layers=4,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=32,
                        learning_rate=0.0001,
                        weight_decay=0.00001 
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                )
            ```
            
        - MLP+TL
            
            ```python
            'ESOL': DatasetConfig(
                    task_type='regression',
                    input_dim=None,
                    output_dim=1,
                    model_config=ModelConfig(
                        dropout_rates= [0.1, 0.1, 0.1],
                        activation='LeakyReLU',
                        use_normalization=True,
                        hidden_dims=[128, 64, 32],
                        num_heads=1,
                        transformer_dim=128,
                        transformer_dropout=0.1,
                        num_layers=2,
                        embedding_dim=16
                    ),
                    training_config=TrainingConfig(
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-05,
                        weight_decay=0.0001 
                    ),
                    use_feature_net=False,
                    requires_imputation=True
                ),
            ```