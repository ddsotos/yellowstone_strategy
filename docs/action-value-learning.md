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

## Continuing Horizon-20 Smoke

Run `action-value-run-010-h20-continuing-s300` used:

```text
source_state_limit = 300
actions_per_state = 0
horizon_learner_turns = 20
continuing_game = true
samples = 8018
```

The continuing collector fixed the sample sparsity problem, but the first
models were worse than the horizon-8 baseline:

```text
self_loss e30, immediate-loss-penalty 1.0:
  p0 loss share ~= 0.430 / 100 games
self_loss e10, immediate-loss-penalty 1.0:
  p0 loss share ~= 0.505 / 100 games
relative_loss e30, immediate-loss-penalty 1.0:
  p0 loss share ~= 0.429 / 100 games
self_loss e30, immediate-loss-penalty 3.0:
  p0 loss share ~= 0.414 / 100 games
self_loss e30, immediate-loss-penalty 5.0:
  p0 loss share ~= 0.417 / 100 games
```

The e30 diagnostics with immediate-loss-penalty `1.0` showed a model two-card
rate of about `0.682`, compared with heuristic's `0.571`. The immediate-loss
penalty helped only slightly. This suggests the current horizon-20 continuing
labels are still encouraging too many costly two-card turns, or that the model
needs a better target scale / policy extraction method before longer horizons
become useful.

## Advantage Gated Policy

For using a learned model as a heuristic improvement layer, the dataset now
stores heuristic-relative labels:

```text
target_self_advantage = heuristic_target_self_loss - action_target_self_loss
target_relative_advantage = heuristic_target_relative_loss - action_target_relative_loss
```

Positive advantage means the candidate action was better than the heuristic
action in the same source state and rollout setting. The evaluator supports:

```bash
python -m yellowstone.action_value_evaluation \
  models/yellowstone_advantage_h8_continuing_self_run001_e30.pt \
  --policy advantage_gated \
  --advantage-margin 6.0 \
  --loss-guard 2 \
  --continuing-learner-turns 200 \
  --games 50
```

Initial run `advantage-run-001-h8-continuing-s500` used:

```text
source_state_limit = 500
actions_per_state = 0
horizon_learner_turns = 8
continuing_game = true
samples = 13196
target = self_advantage
```

Continuing-game 200 learner-turn evaluation:

```text
margin 0.5, loss_guard 2:
  p0 loss share ~= 0.352 / 50 games
margin 1.0, loss_guard 2:
  p0 loss share ~= 0.359 / 50 games
margin 2.0, loss_guard 2:
  p0 loss share ~= 0.341 / 50 games
margin 4.0, loss_guard 2:
  p0 loss share ~= 0.334 / 50 games
margin 6.0, loss_guard 2:
  p0 loss share ~= 0.304 / 50 games
```

The higher margin result is close to the previous h8 greedy model's continuing
loss share around `0.300`, but it still does not beat heuristic-only around
`0.243`. This suggests the first advantage model is not yet accurate enough to
identify beneficial deviations. The next useful data pass should oversample
states where heuristic chooses one card but legal two-card actions exist,
especially low-hand states and no-damage/tolerable-damage two-card choices.
