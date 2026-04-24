"""
Training loop for the RL data balance experiment.

Orchestrates: data mixing -> episode rollout -> reward computation -> logging.
This is a lightweight harness — actual RL optimization depends on your framework
(e.g., TRL, OpenRLHF, custom PPO). This module handles the experiment logic.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..data.mixer import DataMixer, MixedSample
from .reward import RewardFunction, RewardSignal


@dataclass
class EpisodeResult:
    """Result of a single training episode."""
    sample: MixedSample
    test_passed: bool
    flagged_for_human: bool
    reward: RewardSignal
    judge_verdict: Optional[str] = None
    model_output: str = ""
    step: int = 0


@dataclass
class TrainConfig:
    """Training configuration."""
    positive_fraction: float = 0.5
    total_episodes: int = 1000
    batch_size: int = 8
    seed: int = 42
    log_dir: str = "./logs/training"
    # Reward weights
    correct_solve: float = 1.0
    correct_flag: float = 0.8
    rationalization_penalty: float = -1.5
    incorrect_penalty: float = -0.5


class Trainer:
    """
    Manages the training loop for the data balance experiment.

    This is the experiment harness, not a full RL trainer. It:
    1. Mixes data at the configured ratio
    2. Runs episodes (via a pluggable rollout function)
    3. Computes rewards
    4. Logs everything for analysis
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.reward_fn = RewardFunction(
            correct_solve=config.correct_solve,
            correct_flag=config.correct_flag,
            rationalization_penalty=config.rationalization_penalty,
            incorrect_penalty=config.incorrect_penalty,
        )
        self.episode_log: list[dict[str, Any]] = []

    def prepare_data(self, mixer: DataMixer) -> list[MixedSample]:
        """Load and mix training data."""
        mixer.load()
        return mixer.mix(total_size=self.config.total_episodes)

    def compute_episode_reward(
        self,
        sample: MixedSample,
        test_passed: bool,
        flagged_for_human: bool,
        judge_verdict: Optional[str] = None,
        step: int = 0,
    ) -> RewardSignal:
        """Compute reward for a single episode."""
        return self.reward_fn.compute(
            is_impossible=sample.is_impossible,
            test_passed=test_passed,
            flagged_for_human=flagged_for_human,
            judge_verdict=judge_verdict,
        )

    def log_episode(self, result: EpisodeResult) -> None:
        """Log an episode result."""
        entry = {
            "step": result.step,
            "task_id": result.sample.task_id,
            "split": result.sample.split,
            "is_impossible": result.sample.is_impossible,
            "test_passed": result.test_passed,
            "flagged_for_human": result.flagged_for_human,
            "reward_value": result.reward.value,
            "reward_category": result.reward.category,
            "judge_verdict": result.judge_verdict,
            "timestamp": time.time(),
        }
        self.episode_log.append(entry)

    def save_log(self, path: str | None = None) -> str:
        """Save episode log to JSON."""
        if path is None:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            path = str(
                log_dir
                / f"episodes_pf{self.config.positive_fraction:.2f}_{int(time.time())}.json"
            )

        with open(path, "w") as f:
            json.dump(
                {
                    "config": {
                        "positive_fraction": self.config.positive_fraction,
                        "total_episodes": self.config.total_episodes,
                        "reward_config": self.reward_fn.summary(),
                    },
                    "episodes": self.episode_log,
                },
                f,
                indent=2,
            )

        print(f"Saved {len(self.episode_log)} episodes to {path}")
        return path

    def get_running_stats(self) -> dict[str, Any]:
        """Get running statistics from logged episodes."""
        if not self.episode_log:
            return {}

        categories = {}
        total_reward = 0.0
        for ep in self.episode_log:
            cat = ep["reward_category"]
            categories[cat] = categories.get(cat, 0) + 1
            total_reward += ep["reward_value"]

        n = len(self.episode_log)
        rationalization_count = categories.get("rationalization", 0)
        impossible_count = sum(1 for ep in self.episode_log if ep["is_impossible"])

        return {
            "total_episodes": n,
            "mean_reward": total_reward / n,
            "category_counts": categories,
            "rationalization_rate": (
                rationalization_count / impossible_count if impossible_count > 0 else 0.0
            ),
            "correct_flag_rate": (
                categories.get("correct_flag", 0) / impossible_count
                if impossible_count > 0
                else 0.0
            ),
            "solve_rate": (
                categories.get("correct_solve", 0)
                / max(n - impossible_count, 1)
            ),
        }
