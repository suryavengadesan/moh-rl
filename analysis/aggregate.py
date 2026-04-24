"""
Metrics aggregation across data balance configurations.

Loads episode logs and analysis results, computes cross-ratio comparisons.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


class MetricsAggregator:
    """
    Aggregates experiment results across positive fraction sweep values.
    """

    def __init__(self):
        self.episode_data: list[dict[str, Any]] = []
        self.analysis_data: list[dict[str, Any]] = []

    def load_episode_logs(self, log_dir: str) -> "MetricsAggregator":
        """Load all episode log files from a directory."""
        log_path = Path(log_dir)
        for f in sorted(log_path.glob("episodes_pf*.json")):
            with open(f) as fh:
                data = json.load(fh)
                pf = data["config"]["positive_fraction"]
                for ep in data["episodes"]:
                    ep["positive_fraction"] = pf
                    self.episode_data.append(ep)

        print(f"Loaded {len(self.episode_data)} episodes from {log_dir}")
        return self

    def load_analysis_results(self, results_file: str) -> "MetricsAggregator":
        """Load analysis results (from RationalizationAnalyzer)."""
        with open(results_file) as f:
            self.analysis_data = json.load(f)
        print(f"Loaded {len(self.analysis_data)} analysis results")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert episode data to a DataFrame."""
        return pd.DataFrame(self.episode_data)

    def compute_sweep_summary(self) -> pd.DataFrame:
        """
        Compute summary metrics for each positive fraction value.

        Returns:
            DataFrame with one row per positive_fraction, columns for key metrics.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        summary_rows = []
        for pf, group in df.groupby("positive_fraction"):
            n = len(group)
            impossible = group[group["is_impossible"]]
            solvable = group[~group["is_impossible"]]

            rationalization_count = len(
                group[group["reward_category"] == "rationalization"]
            )
            correct_flag_count = len(
                group[group["reward_category"] == "correct_flag"]
            )
            correct_solve_count = len(
                group[group["reward_category"] == "correct_solve"]
            )

            summary_rows.append(
                {
                    "positive_fraction": pf,
                    "total_episodes": n,
                    "n_impossible": len(impossible),
                    "n_solvable": len(solvable),
                    "mean_reward": group["reward_value"].mean(),
                    "rationalization_rate": (
                        rationalization_count / len(impossible)
                        if len(impossible) > 0
                        else 0
                    ),
                    "correct_flag_rate": (
                        correct_flag_count / len(impossible)
                        if len(impossible) > 0
                        else 0
                    ),
                    "solve_rate": (
                        correct_solve_count / len(solvable)
                        if len(solvable) > 0
                        else 0
                    ),
                    "rationalization_count": rationalization_count,
                    "correct_flag_count": correct_flag_count,
                    "correct_solve_count": correct_solve_count,
                }
            )

        return pd.DataFrame(summary_rows).sort_values("positive_fraction")

    def compute_cheating_type_matrix(self) -> pd.DataFrame:
        """
        Build a matrix of cheating type counts per positive fraction.

        Requires analysis_data to be loaded.

        Returns:
            DataFrame with positive_fraction as index, cheating types as columns.
        """
        if not self.analysis_data:
            return pd.DataFrame()

        rows = []
        for entry in self.analysis_data:
            pf = entry.get("positive_fraction")
            ct = entry.get("cheating_type")
            if pf is not None and ct is not None:
                rows.append({"positive_fraction": pf, "cheating_type": ct})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df.pivot_table(
            index="positive_fraction",
            columns="cheating_type",
            aggfunc="size",
            fill_value=0,
        )
