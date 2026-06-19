# Action Value Learning

This path learns finite-horizon action values for turn-level decisions. It is
intended to answer "which card or pair should be played now" while keeping
placement and refill resolution close to the existing heuristic.

The target is an infinite-game approximation:

```text
Q(s, a) ~= future loss over the next N learner turns
```

Lower values are better.

## Dataset

For each learner turn-start state `s`, the collector evaluates turn-level
actions `a`.

The selected action is applied up to the point just before refill randomness.
From there, heuristic players roll forward for a fixed number of learner turns.
The labels are:

```text
target_self_loss = learner additional loss
target_relative_loss = learner additional loss - average player additional loss
```

The additional loss is measured from the original turn-start state, so damage
caused by the selected action is included in the label. Earlier experiments
that measured only after the selected action made damaging two-card actions look
too cheap.

## Features

The model input is:

```text
turn_start_observation + after_action_observation + action_one_hot
```

`after_action_observation` is the learner-perspective observation after the
selected card or pair has been played and before refill randomness is resolved.
This was important because `turn_start_observation + action_one_hot` alone did
not let the model infer the resulting board and hand state accurately enough.

The action space is the existing 36-action turn-level space:

```text
0..5: one-card actions
6..35: ordered two-card actions
```

## Data Collection

Example:

```bash
python -m yellowstone.action_value_dataset \
  --source-games 3000 \
  --source-state-limit 3000 \
  --actions-per-state 0 \
  --horizon-learner-turns 8 \
  --output runs/action-value/samples.jsonl \
  --summary-output runs/action-value/summary.json
```

Use `--actions-per-state 0` to collect all legal turn-level actions for each
source state.

For longer horizons, use continuing-game mode:

```bash
python -m yellowstone.action_value_dataset \
  --source-games 3000 \
  --source-state-limit 3000 \
  --actions-per-state 0 \
  --horizon-learner-turns 20 \
  --continuing-game \
  --output runs/action-value-continuing/samples.jsonl \
  --summary-output runs/action-value-continuing/summary.json
```

Continuing-game mode keeps the normal deck exhaustion settlement but ignores
the 35-point game-over cutoff. When the deck is exhausted, negative cards are
scored, shuffled back into the deck, and play continues. This is for estimating
longer-term loss pressure, not for changing the normal game rules.

## Training

Example:

```bash
python -m yellowstone.action_value_training \
  runs/action-value/samples.jsonl \
  --output-path models/yellowstone_action_value.pt \
  --report-path runs/action-value/training-report.json \
  --target self_loss
```

In the current experiments, `self_loss` produced better greedy play than
`relative_loss`, even when `relative_loss` had lower validation loss.

## Evaluation

Greedy action-value evaluation chooses the legal action with the smallest
predicted value. An additional immediate-loss penalty can be added at evaluation
time:

```bash
python -m yellowstone.action_value_evaluation \
  models/yellowstone_action_value.pt \
  --games 500 \
  --immediate-loss-penalty 1.0 \
  --json-output runs/action-value/evaluation.json
```

Diagnostics compare the greedy policy against the heuristic policy:

```bash
python -m yellowstone.action_value_diagnostics \
  models/yellowstone_action_value.pt \
  --games 100 \
  --immediate-loss-penalty 1.0
```

## Current Findings

- Horizon 20 is too sparse under the current finite 35-point game ending. Many
  samples cannot reach 20 learner turns. Use `--continuing-game` when collecting
  horizon-20 or longer action-value data.
- Horizon 12 improved after adding immediate loss and after-action observation,
  but was still weaker than the best PPO baseline.
- Horizon 8 with heuristic-source states and `self_loss` is currently the most
  promising setting.
- The best current policy is:

```text
models/yellowstone_action_value_h8_heuristic_self_run009_e20.pt
evaluation immediate-loss-penalty = 1.0
500-game p0 loss share ~= 0.2683
```

This is better than the earlier learned PPO result around `0.33`, but still
behind heuristic-only play around `0.24`. It is close enough to justify further
work on action-value data quality and continuing-game targets.

Validation loss is not perfectly aligned with greedy policy quality. Keep
evaluating checkpoints directly instead of selecting only by validation loss.
