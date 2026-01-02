from typing import List

# =========================
# Device
# =========================
DEVICE: str = "cuda"

# =========================
# Embeddings
# =========================
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH_SIZE: int = 256

# =========================
# Temporal split
# =========================
TRAIN_RATIO: float = 0.75
VAL_RATIO: float = 0.15

# =========================
# Stage-0 AutoEncoder
# =========================
AE_COMPRESSION_FACTOR: int = 8
AE_HIDDEN_MULTIPLIERS: List[int] = [4, 2]
AE_DROPOUT: float = 0.1
AE_LR: float = 1e-3
AE_BATCH_SIZE: int = 256
AE_EPOCHS: int = 30

# =========================
# MoE-VAE
# =========================
MOE_INPUT_DIM: int = 96
ENCODER_HIDDEN_DIMS: List[int] = [128, 64, 32, 16]
DECODER_HIDDEN_DIMS: List[int] = [16, 32, 64, 128]
LATENT_DIM: int = 64
NUM_EXPERTS: int = 128
TOP_K: int = 2

MOE_BATCH_SIZE: int = 256
MOE_EPOCHS: int = 50
MOE_LR: float = 3e-4

# =========================
# Loss weights
# =========================
BETA_KL: float = 0.1
ROUTER_ENTROPY_WEIGHT: float = 0.01
LOAD_BALANCE_WEIGHT: float = 0.1
EPS: float = 1e-9
