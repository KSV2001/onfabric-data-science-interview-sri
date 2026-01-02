from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Global hyperparameters for Dynamic Thread Model.
    """

    # Thread model
    K: int = 50
    embed_dim: int = 384
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048

    # Time encoding
    time_features: int = 64

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 30
    gradient_clip: float = 1.0

    # Loss weights
    recon_weight: float = 1.0
    kl_weight: float = 0.1
    temporal_weight: float = 0.05
    sparsity_weight: float = 0.01

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Signal filtering
    signal_threshold: float = 0.3
