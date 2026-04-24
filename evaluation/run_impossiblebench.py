"""
Evaluation wrapper around ImpossibleBench.

Runs a trained model against all splits and collects results
for rationalization analysis.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Add impossiblebench to path if needed
_IMPOSSIBLEBENCH_SRC = Path(__file__).resolve().parent.parent.parent / "impossiblebench" / "src"
if _IMPOSSIBLEBENCH_SRC.exists():
    sys.path.insert(0, str(_IMPOSSIBLEBENCH_SRC))

try:
    from inspect_ai import eval as inspect_eval
    from impossiblebench import impossible_livecodebench
except ImportError:
    inspect_eval = None
    impossible_livecodebench = None


def run_evaluation(
    model: str,
    splits: list[str] | None = None,
    agent_type: str = "minimal",
    max_attempts: int = 3,
    limit: int | None = 50,
    message_limit: int = 30,
    sandbox: str = "docker",
    log_dir: str = "./logs/eval",
    custom_id: str | None = None,
) -> dict[str, Any]:
    """
    Run ImpossibleBench evaluation across specified splits.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o").
        splits: Which splits to evaluate. Defaults to all three.
        agent_type: "minimal" or "tools".
        max_attempts: Max attempts per task.
        limit: Max samples per split.
        message_limit: Max messages per task.
        sandbox: Sandbox type.
        log_dir: Where to store inspect logs.
        custom_id: Optional custom identifier for this run.

    Returns:
        Dict mapping split name to evaluation results metadata.
    """
    if inspect_eval is None or impossible_livecodebench is None:
        raise ImportError(
            "inspect_ai and impossiblebench must be installed. "
            "Run: pip install inspect-ai && pip install -e ../impossiblebench"
        )

    splits = splits or ["original", "oneoff", "conflicting"]
    results = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Evaluating split: {split} | model: {model}")
        print(f"{'='*60}")

        task = impossible_livecodebench(
            split=split,
            agent_type=agent_type,
            max_attempts=max_attempts,
            limit=limit,
            message_limit=message_limit,
            sandbox=sandbox,
            custom_id=custom_id,
        )

        eval_results = inspect_eval(
            task,
            model=model,
            log_dir=log_dir,
            fail_on_error=False,
        )

        results[split] = {
            "model": model,
            "split": split,
            "agent_type": agent_type,
            "limit": limit,
            "log_dir": log_dir,
        }

    return results


def run_sweep(
    model: str,
    positive_fractions: list[float] | None = None,
    **eval_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Run evaluation for multiple data balance configurations.

    This evaluates the same model across all splits for each fraction,
    tagging results with the fraction for later analysis.

    Args:
        model: Model identifier.
        positive_fractions: List of positive fractions to tag results with.
        **eval_kwargs: Passed to run_evaluation.

    Returns:
        Dict mapping fraction label to evaluation results.
    """
    positive_fractions = positive_fractions or [0.95, 0.8, 0.6, 0.5, 0.4, 0.2]
    sweep_results = {}

    for frac in positive_fractions:
        label = f"pf_{frac:.2f}"
        print(f"\n{'#'*60}")
        print(f"Sweep: positive_fraction={frac:.2f}")
        print(f"{'#'*60}")

        sweep_results[label] = run_evaluation(
            model=model,
            custom_id=label,
            **eval_kwargs,
        )

    return sweep_results
