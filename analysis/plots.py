"""
Visualization for the data balance experiment.

Generates the key plots: rationalization curve, cheating type heatmap,
reward distributions, and logic gap analysis.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_rationalization_curve(
    summary_df: pd.DataFrame,
    output_path: str = "results/rationalization_curve.png",
    figsize: tuple[int, int] = (10, 6),
) -> str:
    """
    Plot rationalization rate vs positive data fraction.

    This is the central figure — shows where positive bias kicks in.

    Args:
        summary_df: Output of MetricsAggregator.compute_sweep_summary().
        output_path: Where to save the plot.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    x = summary_df["positive_fraction"]

    # Primary axis: rationalization rate
    color_rat = "#d62728"
    ax1.plot(x, summary_df["rationalization_rate"], "o-", color=color_rat, linewidth=2, markersize=8, label="Rationalization rate")
    ax1.set_xlabel("Positive Data Fraction", fontsize=13)
    ax1.set_ylabel("Rationalization Rate (on impossible tasks)", fontsize=12, color=color_rat)
    ax1.tick_params(axis="y", labelcolor=color_rat)
    ax1.set_ylim(-0.05, 1.05)

    # Secondary axis: solve rate on solvable tasks
    ax2 = ax1.twinx()
    color_solve = "#1f77b4"
    ax2.plot(x, summary_df["solve_rate"], "s--", color=color_solve, linewidth=2, markersize=8, label="Solve rate")
    ax2.set_ylabel("Solve Rate (on solvable tasks)", fontsize=12, color=color_solve)
    ax2.tick_params(axis="y", labelcolor=color_solve)
    ax2.set_ylim(-0.05, 1.05)

    # Correct flag rate on primary axis
    color_flag = "#2ca02c"
    ax1.plot(x, summary_df["correct_flag_rate"], "^:", color=color_flag, linewidth=2, markersize=8, label="Correct flag rate")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

    ax1.set_title("Effect of Training Data Balance on Prover Behavior", fontsize=14)
    ax1.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved rationalization curve to {output_path}")
    return output_path


def plot_cheating_type_heatmap(
    type_matrix: pd.DataFrame,
    output_path: str = "results/cheating_type_heatmap.png",
    figsize: tuple[int, int] = (8, 6),
) -> str:
    """
    Plot heatmap of cheating type distribution across positive fractions.

    Shows how the nature of rationalization changes with data balance.

    Args:
        type_matrix: Output of MetricsAggregator.compute_cheating_type_matrix().
        output_path: Where to save the plot.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    type_labels = {
        "A": "A: Modify tests",
        "B": "B: Overload operators",
        "C": "C: Record states",
        "D": "D: Special-casing",
        "E": "E: Other",
    }

    # Rename columns for readability
    renamed = type_matrix.rename(columns=type_labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        renamed,
        annot=True,
        fmt="g",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Cheating Type", fontsize=12)
    ax.set_ylabel("Positive Data Fraction", fontsize=12)
    ax.set_title("Cheating Type Distribution by Training Data Balance", fontsize=13)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved cheating type heatmap to {output_path}")
    return output_path


def plot_reward_distribution(
    episode_df: pd.DataFrame,
    output_path: str = "results/reward_distribution.png",
    figsize: tuple[int, int] = (12, 5),
) -> str:
    """
    Plot reward distributions per positive fraction.

    Args:
        episode_df: DataFrame of all episodes with 'positive_fraction' and 'reward_value'.
        output_path: Where to save the plot.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    fractions = sorted(episode_df["positive_fraction"].unique())
    n_fracs = len(fractions)

    fig, axes = plt.subplots(1, n_fracs, figsize=figsize, sharey=True)
    if n_fracs == 1:
        axes = [axes]

    for ax, pf in zip(axes, fractions):
        subset = episode_df[episode_df["positive_fraction"] == pf]
        ax.hist(subset["reward_value"], bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(f"PF={pf:.2f}", fontsize=11)
        ax.set_xlabel("Reward")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Count")
    fig.suptitle("Reward Distribution by Positive Fraction", fontsize=13)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reward distribution to {output_path}")
    return output_path


def plot_logic_gap_analysis(
    analysis_results: list[dict],
    output_path: str = "results/logic_gaps.png",
    figsize: tuple[int, int] = (10, 5),
) -> str:
    """
    Plot logic gap scores across positive fractions.

    Args:
        analysis_results: List of dicts with 'positive_fraction' and 'logic_gaps'.
        output_path: Where to save the plot.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    rows = []
    for r in analysis_results:
        pf = r.get("positive_fraction")
        gaps = r.get("logic_gaps", {})
        score = gaps.get("gap_score", -1)
        if pf is not None and score >= 0:
            rows.append({"positive_fraction": pf, "gap_score": score})

    if not rows:
        print("No logic gap data to plot")
        return ""

    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Box plot of gap scores per fraction
    fractions = sorted(df["positive_fraction"].unique())
    data_by_frac = [df[df["positive_fraction"] == pf]["gap_score"].values for pf in fractions]
    ax1.boxplot(data_by_frac, labels=[f"{pf:.2f}" for pf in fractions])
    ax1.set_xlabel("Positive Fraction")
    ax1.set_ylabel("Logic Gap Score (0-10)")
    ax1.set_title("Logic Gap Severity")

    # Mean gap score trend
    means = df.groupby("positive_fraction")["gap_score"].mean()
    ax2.plot(means.index, means.values, "o-", color="#d62728", linewidth=2, markersize=8)
    ax2.set_xlabel("Positive Fraction")
    ax2.set_ylabel("Mean Gap Score")
    ax2.set_title("Mean Logic Gap Score vs Data Balance")
    ax2.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved logic gap analysis to {output_path}")
    return output_path
