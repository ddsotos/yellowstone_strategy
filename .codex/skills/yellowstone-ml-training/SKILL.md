---
name: yellowstone-ml-training
description: Yellowstone Strategy reinforcement-learning runbook. Use when the user asks to run training, evaluate learned models, compare checkpoints, diagnose RL results, choose timesteps for a time budget, or summarize ML experiment outcomes in this repository. Prioritizes fast feedback, preserving partial progress, and evaluating existing checkpoints instead of restarting runs after timeouts.
---

# Yellowstone ML Training

Use this skill for reinforcement-learning experiments in this repository.

## Core Rules

- Do not throw away partial training progress. If a run times out or is interrupted, first look for checkpoints and evaluate the latest usable checkpoint.
- Do not restart from scratch merely to produce a completed model inside the time budget. Restart only when the user explicitly asks for a fresh run, no checkpoint exists, or the partial run produced no usable artifact.
- For fast feedback, prefer checkpoint evaluation over repeating training.
- Treat `models/` and `runs/` as generated artifacts. They are ignored by Git. Summarize results in docs only when the user asks for a committed record.
- Report both training execution status and policy quality. "Training ran" does not mean "the policy improved."
- For the current experiment phase, skip `game_over` completion checks unless the user asks for them or results look structurally impossible. Focus feedback speed on the requested policy metric.
- Prefer each player's loss share over raw average loss: for each game, compute `player_loss / sum(all_player_losses)`, then average that share across games.

## Before Running Training

1. Check `git status --short`.
2. Confirm RL dependencies are installed:
   ```powershell
   .venv\Scripts\python -c "import gymnasium, numpy, stable_baselines3, sb3_contrib, torch, tensorboard"
   ```
3. Run tests if code changed:
   ```powershell
   .venv\Scripts\python -m pytest
   ```
4. Use a new run directory and model name. Do not overwrite previous runs unless the user asks.

Recommended naming:

```text
runs/training-run-###-turn-<steps>
models/yellowstone_turn_maskable_ppo_<steps>
models/checkpoints-turn-<steps>
```

Set `MPLCONFIGDIR` inside the run directory to avoid user-profile matplotlib cache errors.

## Running With A Time Budget

Prefer commands that create checkpoints early and often. If the user gives a short time budget, set `checkpoint_freq` small enough to guarantee a checkpoint before the budget expires.

Example:

```powershell
$runDir = Join-Path (Resolve-Path .).Path 'runs\training-run-004-turn-budget'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$env:MPLCONFIGDIR = Join-Path $runDir 'mpl'
.venv\Scripts\python -m yellowstone.training `
  --total-timesteps 60000 `
  --eval-freq 5000 `
  --eval-episodes 3 `
  --checkpoint-freq 2048 `
  --output-path models\yellowstone_turn_maskable_ppo_60k `
  --checkpoint-dir models\checkpoints-turn-60k `
  --log-dir $runDir `
  --report-path (Join-Path $runDir 'training-report.json') `
  --verbose 1
```

If the command times out, continue with "After Timeout Or Interruption".

## After Timeout Or Interruption

1. Do not rerun from scratch.
2. List checkpoint files:
   ```powershell
   Get-ChildItem models\checkpoints* -Recurse -Filter *.zip | Sort-Object LastWriteTime
   ```
3. Pick the newest checkpoint from the interrupted run.
4. Evaluate it:
   ```powershell
   .venv\Scripts\python -m yellowstone.model_evaluation <checkpoint.zip> --games 20 --json-output <runDir>\checkpoint-eval.json --csv-output <runDir>\checkpoint-eval.csv
   ```
5. Report that the model is partial, with its checkpoint step if visible from the filename.

If the interrupted run produced no checkpoint but did produce logs, summarize the last logged `total_timesteps`, `ep_rew_mean`, and fps. Only then consider a smaller fresh run.

## Evaluation Standards

For learned-vs-heuristic experiments, report at least:

- model path or checkpoint path
- matches
- player 0 win rate
- player 0 average loss share
- average turns
- baseline context, but do not claim "worse than random" unless the random baseline used the same opponents

Preferred direct comparison:

```text
learned player 0 + heuristic players 1-3
random player 0 + heuristic players 1-3
heuristic player 0 + heuristic players 1-3
```

The current `model_evaluation` baseline includes `heuristic_only` and `random_only`; treat those as rough context, not direct head-to-head comparisons.

## Interpreting Results

- Judge policy quality primarily by player 0 loss share against heuristic players. Lower is better; `0.25` is parity in a four-player game by loss share, below `0.25` means player 0 is taking less than its equal share of total losses.
- Raw average loss is secondary context only; it can move with game length and scoring scale.
- If fps drops sharply, inspect action mask generation and environment step cost before running longer.
- For turn-level learning, action space should be 36. If fps is very low, verify `legal_turn_action_mask` is not resolving all 36 actions.

## Current Project Assumptions

- Training uses `YellowstoneTurnGymEnv`, where the learner chooses which hand slot(s) to play.
- Placement, frame choice, and refill source are resolved by heuristic rules.
- Turn-level actions:
  - 6 one-card actions
  - 30 two-card ordered-pair actions
  - 36 total actions
- Generated models and run logs stay in `models/` and `runs/`, which are ignored by Git.
