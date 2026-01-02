import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embed_events(
    events: List[Dict],
    model: SentenceTransformer,
    batch_size: int
) -> np.ndarray:
    """
    Embed event descriptions.

    Args:
        events: list of event dicts with 'description'
        model: sentence transformer
        batch_size: encoding batch size

    Returns:
        Embedding matrix (N, D)
    """
    texts = [e["description"] for e in events]
    out = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        out.append(emb)

    return np.vstack(out)
