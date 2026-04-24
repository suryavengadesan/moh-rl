"""
Data mixer for controlling positive:negative training data balance.

Loads ImpossibleBench splits and creates mixed datasets at configurable ratios.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Dataset, concatenate_datasets


@dataclass
class MixedSample:
    """A training sample with its provenance."""
    task_id: str
    prompt: str
    test: str
    entry_point: str
    split: str  # "original", "oneoff", "conflicting"
    is_impossible: bool
    impossible_type: Optional[str] = None


class DataMixer:
    """
    Controls the balance of positive (solvable) vs negative (impossible)
    training examples from ImpossibleBench.
    """

    HF_DATASET = "fjzzq2002/impossible_livecodebench"

    def __init__(
        self,
        positive_fraction: float = 0.5,
        positive_split: str = "original",
        negative_splits: list[str] | None = None,
        max_samples_per_split: int | None = None,
        seed: int = 42,
    ):
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError(f"positive_fraction must be in [0, 1], got {positive_fraction}")

        self.positive_fraction = positive_fraction
        self.positive_split = positive_split
        self.negative_splits = negative_splits or ["conflicting", "oneoff"]
        self.max_samples_per_split = max_samples_per_split
        self.seed = seed
        self.rng = random.Random(seed)

        self._positive_data: list[MixedSample] = []
        self._negative_data: list[MixedSample] = []

    def load(self) -> "DataMixer":
        """Load data from HuggingFace."""
        # Load positive (solvable) examples
        pos_ds = load_dataset(self.HF_DATASET, split=self.positive_split)
        self._positive_data = [
            MixedSample(
                task_id=row["task_id"],
                prompt=row["prompt"],
                test=row["test"],
                entry_point=row["entry_point"],
                split=self.positive_split,
                is_impossible=False,
            )
            for row in pos_ds
        ]

        # Load negative (impossible) examples
        self._negative_data = []
        for neg_split in self.negative_splits:
            neg_ds = load_dataset(self.HF_DATASET, split=neg_split)
            for row in neg_ds:
                self._negative_data.append(
                    MixedSample(
                        task_id=row["task_id"],
                        prompt=row["prompt"],
                        test=row["test"],
                        entry_point=row["entry_point"],
                        split=neg_split,
                        is_impossible=True,
                        impossible_type=row.get("impossible_type", neg_split),
                    )
                )

        if self.max_samples_per_split:
            self._positive_data = self._positive_data[: self.max_samples_per_split]
            self._negative_data = self._negative_data[: self.max_samples_per_split]

        print(
            f"Loaded {len(self._positive_data)} positive, "
            f"{len(self._negative_data)} negative samples"
        )
        return self

    def mix(self, total_size: int | None = None) -> list[MixedSample]:
        """
        Create a mixed dataset at the configured positive:negative ratio.

        Args:
            total_size: Target dataset size. If None, uses all available data
                        scaled to maintain the ratio.

        Returns:
            Shuffled list of MixedSample at the target ratio.
        """
        if not self._positive_data or not self._negative_data:
            raise RuntimeError("Call .load() first")

        if total_size is None:
            # Use the smaller pool scaled by its fraction to determine total
            max_pos = len(self._positive_data)
            max_neg = len(self._negative_data)
            if self.positive_fraction > 0 and (1 - self.positive_fraction) > 0:
                total_from_pos = int(max_pos / self.positive_fraction)
                total_from_neg = int(max_neg / (1 - self.positive_fraction))
                total_size = min(total_from_pos, total_from_neg)
            elif self.positive_fraction == 1.0:
                total_size = max_pos
            else:
                total_size = max_neg

        n_positive = int(total_size * self.positive_fraction)
        n_negative = total_size - n_positive

        # Sample with replacement if needed
        pos_samples = self._sample(self._positive_data, n_positive)
        neg_samples = self._sample(self._negative_data, n_negative)

        mixed = pos_samples + neg_samples
        self.rng.shuffle(mixed)

        print(
            f"Mixed dataset: {n_positive} positive ({self.positive_fraction:.0%}), "
            f"{n_negative} negative ({1 - self.positive_fraction:.0%}), "
            f"{len(mixed)} total"
        )
        return mixed

    def _sample(self, pool: list[MixedSample], n: int) -> list[MixedSample]:
        """Sample n items from pool, with replacement if n > len(pool)."""
        if n <= len(pool):
            return self.rng.sample(pool, n)
        # Oversample with replacement
        return [self.rng.choice(pool) for _ in range(n)]

    @property
    def positive_count(self) -> int:
        return len(self._positive_data)

    @property
    def negative_count(self) -> int:
        return len(self._negative_data)
