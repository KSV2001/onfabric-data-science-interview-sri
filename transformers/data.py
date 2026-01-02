import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


class SearchHistoryDataset(Dataset):
    """
    Dataset for temporally ordered search events.
    """

    def __init__(
        self,
        events: List[Dict],
        embedder,
        max_seq_len: int = 256,
    ):
        self.events = sorted(events, key=lambda e: e["timestamp"])
        self.embedder = embedder
        self.max_seq_len = max_seq_len

        self.embeddings = embedder.encode(
            [e["description"] for e in self.events],
            convert_to_tensor=True,
        )

        self.timestamps = torch.tensor(
            [e["timestamp"].timestamp() for e in self.events],
            dtype=torch.float32,
        )

        self.deltas = torch.zeros_like(self.timestamps)
        self.deltas[1:] = self.timestamps[1:] - self.timestamps[:-1]

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embeddings[idx],
            "timestamp": self.timestamps[idx],
            "delta": self.deltas[idx],
        }
