from typing import List, Dict, Tuple
from .time_utils import normalize_timestamp


def split_by_time(
    events: List[Dict],
    train_ratio: float,
    val_ratio: float
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Chronologically split events into train/val/test.

    Args:
        events: list of event dicts containing 'timestamp'
        train_ratio: fraction for training
        val_ratio: fraction for validation

    Returns:
        train, val, test event lists
    """
    enriched = [
        {**e, "ts_norm": normalize_timestamp(e["timestamp"])}
        for e in events
    ]
    enriched.sort(key=lambda x: x["ts_norm"])

    n = len(enriched)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    return (
        enriched[:n_train],
        enriched[n_train:n_train + n_val],
        enriched[n_train + n_val:]
    )
