"""
Baseline human-aligned user modeling approach.

Pipeline:
    Events → Episodes → Themes

This is an intentionally simple model, similar to the "dummy model" mentioned in the challenge description.

This baseline uses:
- Sentence-level embeddings
- Temporal continuity
- Online clustering heuristics

The design emphasizes interpretability and extensibility
over aggressive optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import (
    DEVICE,
    EMBEDDING_MODEL,
    EPISODE_SIM_THRESHOLD,
    EPISODE_TIME_GAP_HOURS,
    MIN_EPISODE_SIZE_FOR_THEME,
    THEME_SIM_THRESHOLD,
    TRAIN_FRAC,
    VAL_FRAC,
)

# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------

@dataclass
class Event:
    """
    Atomic user interaction unit.
    """
    idx: int
    timestamp: float
    description: str
    embedding: np.ndarray | None = None
    episode_id: int | None = None
    theme_id: int | None = None


@dataclass
class Episode:
    """
    Temporally contiguous group of semantically similar events.
    """
    id: int
    events: List[int]
    centroid: np.ndarray
    sum_vec: np.ndarray
    count: int
    start_time: float
    end_time: float


@dataclass
class Theme:
    """
    Higher-level latent grouping of related episodes.
    """
    id: int
    episode_ids: List[int]
    centroid: np.ndarray
    sum_vec: np.ndarray
    count: int


# ---------------------------------------------------------------------
# MAIN MODEL
# ---------------------------------------------------------------------

class BaselineUserModel:
    """
    Baseline hierarchical user model.

    This class implements a simple, interpretable hierarchy:
        Events → Episodes → Themes
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        device: str = DEVICE,
        episode_sim_threshold: float = EPISODE_SIM_THRESHOLD,
        episode_time_gap_hours: float = EPISODE_TIME_GAP_HOURS,
        theme_sim_threshold: float = THEME_SIM_THRESHOLD,
        min_episode_size_for_theme: int = MIN_EPISODE_SIZE_FOR_THEME,
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model, device=device)

        self.episode_sim_threshold = episode_sim_threshold
        self.episode_time_gap_sec = episode_time_gap_hours * 3600
        self.theme_sim_threshold = theme_sim_threshold
        self.min_episode_size_for_theme = min_episode_size_for_theme

        self.events: List[Event] = []
        self.episodes: Dict[int, Episode] = {}
        self.themes: Dict[int, Theme] = {}

    # -----------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------

    def load_events(self, data: Iterable[Dict[str, Any]]) -> None:
        """
        Load cleaned events into the model.

        Parameters
        ----------
        data : Iterable[Dict[str, Any]]
            Each item must contain 'timestamp' and 'description'.
        """
        sorted_data = sorted(data, key=lambda x: x["timestamp"])
        self.events = [
            Event(
                idx=i,
                timestamp=d["timestamp"],
                description=d["description"],
            )
            for i, d in enumerate(sorted_data)
        ]

    # -----------------------------------------------------------------
    # EMBEDDING
    # -----------------------------------------------------------------

    def embed_events(self, batch_size: int = 128) -> None:
        """
        Compute sentence embeddings for all events.
        """
        texts = [e.description for e in self.events]

        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for event, emb in zip(self.events, embeddings):
            event.embedding = emb

    # -----------------------------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # -----------------------------------------------------------------

    def time_split(self) -> Tuple[List[Event], List[Event], List[Event]]:
        """
        Split events into train / validation / test sets by time.
        """
        times = np.array([e.timestamp for e in self.events])
        t_min, t_max = times.min(), times.max()

        t_train = t_min + TRAIN_FRAC * (t_max - t_min)
        t_val = t_min + (TRAIN_FRAC + VAL_FRAC) * (t_max - t_min)

        train = [e for e in self.events if e.timestamp <= t_train]
        val = [e for e in self.events if t_train < e.timestamp <= t_val]
        test = [e for e in self.events if e.timestamp > t_val]

        return train, val, test

    # -----------------------------------------------------------------
    # EPISODE CONSTRUCTION
    # -----------------------------------------------------------------

    def build_episodes(self) -> None:
        """
        Construct episodes using online temporal clustering.
        """
        self.episodes.clear()

        episode_id = 0
        active_episode: Episode | None = None

        for event in self.events:
            if active_episode is None:
                active_episode = self._start_episode(event, episode_id)
                episode_id += 1
                continue

            sim = float(np.dot(event.embedding, active_episode.centroid))
            time_gap = event.timestamp - active_episode.end_time

            if (
                sim >= self.episode_sim_threshold
                and time_gap <= self.episode_time_gap_sec
            ):
                self._extend_episode(active_episode, event)
            else:
                active_episode = self._start_episode(event, episode_id)
                episode_id += 1

    def _start_episode(self, event: Event, episode_id: int) -> Episode:
        ep = Episode(
            id=episode_id,
            events=[event.idx],
            centroid=event.embedding.copy(),
            sum_vec=event.embedding.copy(),
            count=1,
            start_time=event.timestamp,
            end_time=event.timestamp,
        )
        event.episode_id = episode_id
        self.episodes[episode_id] = ep
        return ep

    def _extend_episode(self, episode: Episode, event: Event) -> None:
        episode.events.append(event.idx)
        episode.sum_vec += event.embedding
        episode.count += 1
        episode.centroid = episode.sum_vec / episode.count
        episode.end_time = event.timestamp
        event.episode_id = episode.id

    # -----------------------------------------------------------------
    # THEME CONSTRUCTION
    # -----------------------------------------------------------------

    def build_themes(self) -> None:
        """
        Cluster episodes into higher-level themes.
        """
        self.themes.clear()

        theme_id = 0
        theme_centroids: List[np.ndarray] = []
        theme_ids: List[int] = []

        for episode in self.episodes.values():
            if episode.count < self.min_episode_size_for_theme:
                continue

            if not theme_centroids:
                self._create_theme(episode, theme_id)
                theme_centroids.append(self.themes[theme_id].centroid)
                theme_ids.append(theme_id)
                theme_id += 1
                continue

            sims = np.dot(np.stack(theme_centroids), episode.centroid)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.theme_sim_threshold:
                theme = self.themes[theme_ids[best_idx]]
                theme.episode_ids.append(episode.id)
                theme.sum_vec += episode.centroid
                theme.count += 1
                theme.centroid = theme.sum_vec / theme.count
                theme_centroids[best_idx] = theme.centroid
            else:
                self._create_theme(episode, theme_id)
                theme_centroids.append(self.themes[theme_id].centroid)
                theme_ids.append(theme_id)
                theme_id += 1

        self._propagate_theme_ids()

    def _create_theme(self, episode: Episode, theme_id: int) -> None:
        self.themes[theme_id] = Theme(
            id=theme_id,
            episode_ids=[episode.id],
            centroid=episode.centroid.copy(),
            sum_vec=episode.centroid.copy(),
            count=1,
        )

    def _propagate_theme_ids(self) -> None:
        for theme in self.themes.values():
            for ep_id in theme.episode_ids:
                for ev_id in self.episodes[ep_id].events:
                    self.events[ev_id].theme_id = theme.id

    # -----------------------------------------------------------------
    # TRAINING PIPELINE
    # -----------------------------------------------------------------

    def train(self) -> None:
        """
        Full training pipeline.
        """
        self.embed_events()
        self.build_episodes()
        self.build_themes()

    # -----------------------------------------------------------------
    # QUERY INTERFACE
    # -----------------------------------------------------------------

    def get_events_by_theme(self, theme_id: int) -> List[Event]:
        return [e for e in self.events if e.theme_id == theme_id]

    def get_events_by_episode(self, episode_id: int) -> List[Event]:
        return [self.events[i] for i in self.episodes[episode_id].events]
