"""Generic normalization-stats accumulator.

Maintains RunningStats (mean/std via Welford's) and FeatureQuantileTracker
(q01/q99) for a configurable set of named feature keys, plus their _diff
counterparts computed via chunk-delta over a fixed horizon.

Uses stats_utils from robocoin/utils/.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'robocoin'))

from utils.stats_utils import (
    RunningStats,
    FeatureQuantileTracker,
    update_running_from_array,
    update_running_stats,
    compute_chunk_delta_stats,
    running_to_entry,
)


@dataclass
class FeatureSpec:
    """Describes one feature to track."""
    key: str
    dim: int
    names: List[str]
    track_diff: bool = True
    diff_key: str = field(default = '', init = False)

    def __post_init__(self):
        if self.track_diff:
            base = self.key.split('.')[-1] if '.' in self.key else self.key
            self.diff_key = f'{base}_diff'


class NormStatsAccumulator:
    """Accumulates mean/std/q01/q99 for a set of features and their chunk-delta diffs."""

    def __init__(self, specs: List[FeatureSpec], quantile_keep: int = 2000, diff_horizon: int = 50):
        self.specs = specs
        self.quantile_keep = quantile_keep
        self.diff_horizon = diff_horizon
        self.num_episodes = 0
        self.num_steps = 0

        self._running: Dict[str, RunningStats] = {}
        self._quantiles: Dict[str, FeatureQuantileTracker] = {}

        for spec in specs:
            self._running[spec.key] = RunningStats()
            self._quantiles[spec.key] = FeatureQuantileTracker(
                (spec.dim,), max_size = quantile_keep,
            )
            if spec.track_diff:
                self._running[spec.diff_key] = RunningStats()
                self._quantiles[spec.diff_key] = FeatureQuantileTracker(
                    (diff_horizon, spec.dim), max_size = quantile_keep,
                )

    def update(self, arrays: Dict[str, np.ndarray]) -> None:
        """Feed one episode's arrays. Diffs are computed internally via chunk-delta."""
        self.num_episodes += 1
        self.num_steps += arrays[self.specs[0].key].shape[0]

        for spec in self.specs:
            values = arrays[spec.key]
            update_running_from_array(self._running[spec.key], values)
            self._quantiles[spec.key].update(values)

            if spec.track_diff:
                chunk_stats = compute_chunk_delta_stats(
                    values,
                    quantiles = self._quantiles[spec.diff_key],
                    horizon = self.diff_horizon,
                )
                update_running_stats(self._running[spec.diff_key], chunk_stats)

    def finalize(self) -> Dict:
        result = {
            'num_episodes': self.num_episodes,
            'num_steps': self.num_steps,
        }
        for spec in self.specs:
            result[spec.key] = running_to_entry(
                self._running[spec.key],
                self._quantiles[spec.key],
                spec.names,
            )
            if spec.track_diff:
                result[spec.diff_key] = running_to_entry(
                    self._running[spec.diff_key],
                    self._quantiles[spec.diff_key],
                    spec.names,
                    include_count = True,
                    first_row_quantile_bounds = (-1.0, 1.0),
                )
        return result
