# Action Value Learning

This path is for learning finite-horizon action values instead of using only
`V(after_state)` reward shaping.

The target is an infinite-game approximation:

```text
Q(s, a) ~= future relative loss over the next N learner turns
```

For each learner turn-start state `s`, the collector samples legal turn-level
actions `a`, applies the selected cards and stops before refill randomness when
refill is required. From that point, heuristic play rolls forward for a fixed
number of learner turns. The labels are:

```text
target_self_loss = learner_future_loss
target_relative_loss = learner_future_loss - average_player_future_loss
```

Lower values are better.

## Data Collection

```bash
python -m yellowstone.action_value_dataset \
  --source-games 300 \
  --source-state-limit 1000 \
  --actions-per-state 6 \
  --horizon-learner-turns 20 \
  --exploratory-sources \
  --one-card-probability 6=0.8 \
  --one-card-probability 5=0.6 \
  --one-card-probability 4=0.4 \
  --output runs/action-value/samples.jsonl \
  --summary-output runs/action-value/summary.json
```

The input features are the normalized turn-start observation plus a one-hot
turn-level action index. The action space is the existing 36-action turn-level
space:

```text
0..5: one-card actions
6..35: ordered two-card actions
```

## Training

```bash
python -m yellowstone.action_value_training \
  runs/action-value/samples.jsonl \
  --output-path models/yellowstone_action_value.pt \
  --report-path runs/action-value/training-report.json \
  --target relative_loss
```

Use `relative_loss` first. It directly represents whether the learner is losing
more or less than the table average over the future horizon. `self_loss` is also
available when absolute future loss is the desired label.

## Initial Smoke Result

The first smoke run used 30 source games, at most 80 source states, 4 sampled
actions per state, and a 6 learner-turn horizon. It produced 240 samples and a
small model successfully. This verifies the pipeline, not model quality.

Next useful checks:

- collect a larger horizon-20 dataset;
- compare validation loss for `relative_loss` and `self_loss`;
- add a greedy action-value policy evaluator before using the model as an RL
  reward.
