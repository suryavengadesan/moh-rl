# MOH-RL 🌸: A Framework for Mitigating Reward Hacking

MOH-RL (Mitigating Optimization-induced Hacking with RL) is a framework for studying rationalization behavior in LLMs — how models learn to cheat, when optimization pressure amplifies it, and what interventions can reduce it.

Built on [ImpossibleBench](https://arxiv.org/abs/2510.20270) for task generation and cheating detection, and [rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) for GRPO training infrastructure.

📄 **[Read the paper write-up draft on Overleaf](https://www.overleaf.com/read/ggnrqcnjrbsx#193cce)**

## Research Directions

This framework is designed to support multiple lines of investigation:

**Direction 1: RL amplification of rationalization**
Base models already cheat on impossible tasks (ImpossibleBench shows this). Does RL training amplify it? By training with varying positive:negative data ratios and measuring cheating rates, we can map how optimization pressure scales rationalization behavior beyond the baseline.

**Direction 2: Reward shaping as mitigation**
Starting from the baseline cheating rate, can reward shaping during RL training reduce rationalization? This uses the screening and penalty interventions from rl-rewardhacking, applied to impossible task detection rather than loophole exploitation.

**Direction 3: In-context data balance (no fine-tuning)**
Does the ratio of solvable vs impossible examples in the prompt context shift behavior in base models? This tests whether data balance effects are a property of optimization or of exposure, and runs on API models without GPU.

**Direction 4: Cheating strategy evolution**
How does the nature of cheating change under different conditions? Do models shift from simple strategies (special-casing) to sophisticated ones (operator overloading, state recording) as training progresses or as data balance shifts?

## How the RL Training Works

We take a base model (Qwen3-4B) and RL fine-tune it using GRPO to solve coding tasks. The reward is simple: your code passes the tests, you get a positive reward.

The training dataset is a mix of:
- Solvable tasks (ImpossibleBench `original` split) — legitimate problems with correct tests
- Impossible tasks (ImpossibleBench `conflicting`/`oneoff` splits) — problems where the tests contradict the spec, so no correct solution exists

We vary the ratio across runs (e.g., 95/5, 80/20, 50/50). The reward signal is the same for both — "did your code pass?" We are not training the model to detect impossible tasks. We are observing what behavior emerges under different data balances.

The hypothesis: when positive signal dominates, RL optimization pressure pushes the model toward "pass tests at all costs," leading it to cheat on impossible tasks. Lower positive fractions mean the model encounters failure more often, potentially learning different strategies.

## Project Structure

```
cse579/
├── config/default.yaml               # experiment configuration
├── data/
│   ├── mixer.py                      # positive:negative data balance controller
│   └── impossiblebench_prep.py       # convert ImpossibleBench → rl-rewardhacking format
├── training/
│   ├── reward.py                     # custom reward function (subclasses rl-rewardhacking)
│   ├── screening.py                  # screening function for rationalization
│   ├── config.py                     # training config extending GRPOConfig
│   ├── runner.py                     # wraps VerlGRPO to launch training
│   └── train.py                      # post-training episode logging
├── evaluation/
│   ├── evaluator.py                  # CodeEvaluator for ImpossibleBench test format
│   ├── run_impossiblebench.py        # wrapper around ImpossibleBench eval tasks
│   └── rationalization.py            # LLM Judge + logic gap detector
├── analysis/
│   ├── aggregate.py                  # cross-ratio metrics aggregation
│   └── plots.py                      # rationalization curves, heatmaps, distributions
├── scripts/
│   ├── prepare_data.py               # generate mixed JSONL at a given ratio
│   ├── run_training.py               # launch a single training run
│   └── sweep.py                      # full sweep across ratios
├── rl-rewardhacking/                 # git submodule — GRPO training infrastructure
└── impossiblebench/                  # git submodule — impossible task benchmark
```

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd cse579
pip install -r requirements.txt
pip install -e rl-rewardhacking/
pip install -e impossiblebench/
```

GPU required for RL training. Data prep, in-context experiments, and analysis run on CPU or API.

## Usage

### Prepare a dataset at a specific ratio

```bash
python -m scripts.prepare_data --fraction 0.8 --output data/train_pf0.80.jsonl
```

### Run a single training

```bash
python -m scripts.run_training --fraction 0.8 --model qwen/Qwen3-4B --steps 200
```

### Run the full sweep

```bash
python -m scripts.sweep --fractions 0.95 0.8 0.6 0.5 0.4 0.2 --steps 200
```

## How It Works

### Data Mixing

The `DataMixer` loads ImpossibleBench splits from HuggingFace:
- Positive (solvable): `original` split — legitimate coding tasks
- Negative (impossible): `conflicting` and `oneoff` splits — tasks where spec and tests contradict

You set a `positive_fraction` (0.0–1.0) and the mixer produces a shuffled dataset at that ratio. The mixer is used for both RL training data (exported as JSONL for verl) and in-context experiment construction.

### RL Training

Training uses GRPO via the rl-rewardhacking/verl infrastructure:
1. Model generates rollouts (code completions) via vLLM
2. Our custom reward function scores each completion (did the code pass tests?)
3. verl computes GRPO advantages normalized within prompt groups
4. LoRA weights on the base model get updated via policy gradient
5. Repeat for N steps

The reward function does not distinguish between solvable and impossible tasks — it just checks if tests pass. The experiment measures what the model learns to do under that uniform pressure at different data balances.

### Rationalization Detection

Evaluation uses three layers of detection, applicable to both RL-trained models and base model experiments:
1. Binary classification — LLM Judge labels transcripts as "cheating" or "legit"
2. Type classification — categorizes cheating behavior (A: modify tests, B: overload operators, C: record states, D: special-casing, E: other)
3. Logic gap detection — scores reasoning rigor on a 0–10 scale

### Baseline Comparison

All experiments compare against the base model's cheating rate on ImpossibleBench (no fine-tuning). This baseline is critical — it separates "rationalization that was already there" from "rationalization that RL training introduced or amplified."

## Key Outputs

- Rationalization curve — cheating rate vs positive data fraction, with base model baseline
- Cheating type heatmap — how cheating strategies shift across conditions
- Reward distributions — per-condition reward histograms
- Logic gap analysis — reasoning rigor scores across the sweep
- Baseline delta — how much RL training changes cheating rate relative to the base model

## References

- [ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases](https://arxiv.org/abs/2510.20270) (Zhong et al., 2025)
- [Impossible-LiveCodeBench dataset](https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench)
- [rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) — GRPO training infrastructure
