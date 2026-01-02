"""
Search Event cleaning and normalization utilities.

This module converts raw search / activity events into
lightweight, text-based representations suitable for
downstream modeling.
"""

from __future__ import annotations

import re
import json
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from .config import (
    GOOGLE_REDIRECT_DOMAIN,
    GOOGLE_REDIRECT_PATH,
    MAX_DF_RATIO,
    TOKEN_SPLIT_PATTERN,
    URL_REGEX_PATTERN,
)

# ---------------------------------------------------------------------
# REGEXES
# ---------------------------------------------------------------------

_URL_REGEX = re.compile(URL_REGEX_PATTERN)


# ---------------------------------------------------------------------
# TIMESTAMP HANDLING
# ---------------------------------------------------------------------

def parse_timestamp(time_str: str) -> pd.Timestamp:
    """
    Convert an ISO-8601 timestamp string to a UTC pandas Timestamp.

    Parameters
    ----------
    time_str : str
        ISO-8601 formatted timestamp.

    Returns
    -------
    pd.Timestamp
        Parsed timestamp in UTC. Invalid formats return NaT.
    """
    return pd.to_datetime(time_str, utc=True, errors="coerce")


# ---------------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """
    Remove URLs from a text string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with URLs removed.
    """
    return _URL_REGEX.sub("", text)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric tokens.

    Numeric-only tokens are discarded.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of normalized tokens.
    """
    return [
        t
        for t in re.split(TOKEN_SPLIT_PATTERN, text.lower())
        if t and not t.isdigit()
    ]


# ---------------------------------------------------------------------
# URL SEMANTIC EXTRACTION
# ---------------------------------------------------------------------

def resolve_google_redirect(url: str) -> str:
    """
    Resolve Google redirect URLs to their target destination.

    Parameters
    ----------
    url : str
        Input URL.

    Returns
    -------
    str
        Resolved URL if redirect, otherwise original URL.
    """
    if not url:
        return ""

    parsed = urlparse(url)

    if GOOGLE_REDIRECT_DOMAIN in parsed.netloc and parsed.path == GOOGLE_REDIRECT_PATH:
        query = parse_qs(parsed.query).get("q")
        if query:
            return query[0]

    return url


def normalize_url(url: str) -> List[str]:
    """
    Normalize a URL into semantic tokens based on domain and path.

    Query parameters and fragments are ignored.

    Parameters
    ----------
    url : str
        Input URL.

    Returns
    -------
    List[str]
        List of semantic tokens extracted from the URL.
    """
    try:
        url = resolve_google_redirect(url)
        parsed = urlparse(unquote(url))
    except Exception:
        return []

    tokens: List[str] = []

    domain = parsed.netloc.replace("www.", "")
    if domain:
        tokens.append(domain.lower())

    path_tokens = [
        t.lower()
        for t in re.split(r"[-_/]", parsed.path)
        if t and not t.isdigit()
    ]

    tokens.extend(path_tokens)
    return tokens


# ---------------------------------------------------------------------
# EVENT CLEANING
# ---------------------------------------------------------------------

def clean_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a single raw event into a normalized representation.

    The description field combines:
    - Tokenized title text (URLs removed)
    - Semantic URL tokens (from titleUrl only)

    Parameters
    ----------
    event : Dict[str, Any]
        Raw event dictionary.

    Returns
    -------
    Dict[str, Any]
        Cleaned event with timestamp, event type, and description.
    """
    output: Dict[str, Any] = {}

    # Timestamp
    if "time" in event:
        output["timestamp"] = datetime.fromisoformat(
            event["time"].replace("Z", "+00:00")
        )
    else:
        output["timestamp"] = None

    # Event type
    output["event_type"] = event.get("header", "").lower()

    tokens: List[str] = []

    # Title text
    title = remove_urls(event.get("title", ""))
    tokens.extend(tokenize(title))

    # URL semantics (only from titleUrl)
    if event.get("titleUrl"):
        tokens.extend(normalize_url(event["titleUrl"]))

    # Deduplicate tokens while preserving order
    seen = set()
    deduped_tokens = [t for t in tokens if not (t in seen or seen.add(t))]

    output["description"] = " ".join(deduped_tokens)
    return output


def clean_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and normalize a collection of events.

    Parameters
    ----------
    events : Iterable[Dict[str, Any]]
        Raw events.

    Returns
    -------
    List[Dict[str, Any]]
        Cleaned events.
    """
    return [clean_event(e) for e in events]


# ---------------------------------------------------------------------
# DOCUMENT FREQUENCY FILTERING
# ---------------------------------------------------------------------

def compute_document_frequency(docs: Iterable[str]) -> Counter:
    """
    Compute document frequency (DF) over a corpus.

    Parameters
    ----------
    docs : Iterable[str]
        Collection of documents.

    Returns
    -------
    Counter
        Token -> document count mapping.
    """
    df = Counter()
    for doc in docs:
        df.update(set(doc.split()))
    return df


def df_filter_sentence(
    sentence: str,
    df: Counter,
    num_docs: int,
    max_df_ratio: float = MAX_DF_RATIO,
) -> str:
    """
    Filter tokens in a sentence based on document frequency.

    Tokens appearing in more than `max_df_ratio` fraction
    of documents are removed.

    Parameters
    ----------
    sentence : str
        Input sentence.
    df : Counter
        Document frequency counter.
    num_docs : int
        Total number of documents.
    max_df_ratio : float, optional
        Maximum allowed DF ratio, by default MAX_DF_RATIO.

    Returns
    -------
    str
        Filtered sentence.
    """
    tokens = sentence.split()
    kept = [
        t for t in tokens
        if df.get(t, 0) / num_docs <= max_df_ratio
    ]
    return " ".join(kept)
