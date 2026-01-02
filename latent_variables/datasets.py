import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Torch dataset for embedding matrices.
    """

    def __init__(self, emb):
        self.emb = torch.tensor(emb, dtype=torch.float32)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return self.emb[idx]
