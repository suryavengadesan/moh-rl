"""
Main experiment script: run the full data balance sweep.

Usage:
    python -m scripts.sweep                    # run with defaults
    python -m scripts.sweep --fractions 0.9 0.5 0.1 --limit 20
    python -m scripts.sweep --config config/default.yaml
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.mixer import DataMixer
from training.reward import RewardFunction
from training.train import Trainer, TrainConfig
from analysis.aggregate import MetricsAggregator
from analysis.plots import (
    plot_rationalization_curve,
    plot_reward_distribution,
)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_sweep(
    positive_fractions: list[float],
    total_episodes: int = 200,
    seed: int = 42,
    log_dir: str = "./logs/training",
    results_dir: str = "./results",
):
    """
    Run the data balance sweep experiment.

    For each positive fraction:
    1. Mix data at that ratio
    2. Simulate episodes (placeholder for actual RL rollouts)
    3. Compute rewards
    4. Log results

    Then aggregate and plot.
    """
    all_log_files = []

    for pf in positive_fractions:
        print(f"\n{'='*60}")
        print(f"Running sweep: positive_fraction={pf:.2f}")
        print(f"{'='*60}")

        config = TrainConfig(
            positive_fraction=pf,
            total_episodes=total_episodes,
            seed=seed,
            log_dir=log_dir,
        )

        trainer = Trainer(config)

        # Prepare data
        mixer = DataMixer(
            positive_fraction=pf,
            seed=seed,
        )
        samples = trainer.prepare_data(mixer)

        # Simulate episodes
        # NOTE: Replace this with actual RL rollouts against your model.
        # This placeholder demonstrates the logging and analysis pipeline.
        for step, sample in enumerate(samples):
            # --- PLACEHOLDER: replace with actual model inference ---
            # In a real run, you'd:
            # 1. Send the task to the model
            # 2. Get the model's output
            # 3. Run tests to check if they pass
            # 4. Check if the model flagged for human intervention
            import random
            rng = random.Random(seed + step)

            if sample.is_impossible:
                # Simulate: higher positive bias -> more rationalization
                rationalization_prob = 0.1 + 0.7 * pf
                test_passed = rng.random() < rationalization_prob
                flagged = not test_passed and rng.random() < (1 - pf)
            else:
                test_passed = rng.random() < 0.6
                flagged = False
            # --- END PLACEHOLDER ---

            reward = trainer.compute_episode_reward(
                sample=sample,
                test_passed=test_passed,
                flagged_for_human=flagged,
                step=step,
            )

            from training.train import EpisodeResult
            trainer.log_episode(EpisodeResult(
                sample=sample,
                test_passed=test_passed,
                flagged_for_human=flagged,
                reward=reward,
                step=step,
            ))

        # Print running stats
        stats = trainer.get_running_stats()
        print(f"Stats for pf={pf:.2f}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Save log
        log_file = trainer.save_log()
        all_log_files.append(log_file)

    # Aggregate and plot
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}")

    aggregator = MetricsAggregator()
    aggregator.load_episode_logs(log_dir)

    summary = aggregator.compute_sweep_summary()
    print("\nSweep Summary:")
    print(summary.to_string(index=False))

    # Save summary
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    summary.to_csv(f"{results_dir}/sweep_summary.csv", index=False)

    # Plot
    episode_df = aggregator.to_dataframe()
    plot_rationalization_curve(summary, output_path=f"{results_dir}/rationalization_curve.png")
    plot_reward_distribution(episode_df, output_path=f"{results_dir}/reward_distribution.png")

    print(f"\nResults saved to {results_dir}/")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run data balance sweep experiment")
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.95, 0.8, 0.6, 0.5, 0.4, 0.2],
        help="Positive fractions to sweep",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", default="./logs/training", help="Log directory")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    parser.add_argument("--config", default=None, help="YAML config file (overrides other args)")

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        fractions = cfg.get("data", {}).get("sweep_fractions", args.fractions)
        episodes = cfg.get("training", {}).get("max_steps", args.episodes)
        seed = cfg.get("experiment", {}).get("seed", args.seed)
    else:
        fractions = args.fractions
        episodes = args.episodes
        seed = args.seed

    run_sweep(
        positive_fractions=fractions,
        total_episodes=episodes,
        seed=seed,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
