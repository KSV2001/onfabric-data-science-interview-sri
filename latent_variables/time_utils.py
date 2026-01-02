from datetime import datetime
from typing import Union


def normalize_timestamp(ts: Union[int, float, str, datetime]) -> float:
    """
    Convert timestamp to UNIX float seconds.

    Args:
        ts: int, float, ISO string, or datetime

    Returns:
        Normalized timestamp as float
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        return datetime.fromisoformat(ts).timestamp()
    if isinstance(ts, datetime):
        return ts.timestamp()

    raise ValueError("Unsupported timestamp format")
