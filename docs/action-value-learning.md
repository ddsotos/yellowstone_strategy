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

## Binary Two-Card Decision

The next direction is to stop learning "which card or pair to play" and learn
only this binary choice at the start of the learner's turn:

```text
STOP_AFTER_ONE
PLAY_SECOND_HEURISTIC
```

Card choice, frame choice, and refill choice remain heuristic. `STOP_AFTER_ONE`
uses the standard heuristic first card and then ends the turn. `PLAY_SECOND_HEURISTIC`
chooses a two-card heuristic pair from the original turn-start state, so a strong
one-card candidate cannot accidentally block a better two-card line. The
implementation entry point is `yellowstone.two_card_decision`.

The rollout comparison uses `horizon_learner_turns=4` by default.

The first supported diagnostic cases are:

- `HEURISTIC_STOPS_WITH_SECOND_AVAILABLE`: heuristic stops after one card even
  though a second card can legally be played. A learned policy may force the
  two-card heuristic line.
- `HEURISTIC_PLAYS_DAMAGING_SECOND`: the two-card heuristic line increases
  negative cards while also gaining positive score. A learned policy may stop
  instead.

`choose_two_card_decision_by_rollout` compares both choices with the same
finite-horizon heuristic rollout. This is an oracle-style method for dataset
generation and diagnostics; it is not intended to replace the learned model.

The observation now also includes the previous turn play counts for the three
opponents, in relative order from the current player.

For the binary two-card decision, the compact board features are:

```text
rank_counts[7]
left_column_index
left_center_right_column_counts[3]
```

The observation also includes two immediate heuristic bonus features:

```text
one_card_heuristic_bonus
two_card_heuristic_bonus
```

These bonuses are computed from the same reusable heuristic turn plans used by
the binary decision resolver, so the model input and decision execution share
the same one-card/two-card assumptions.

### Binary Decision Dataset

Use `yellowstone.two_card_decision_dataset` to generate supervised labels for
the binary policy:

```bash
python -m yellowstone.two_card_decision_dataset \
  --source-games 1000 \
  --source-state-limit 1000 \
  --max-source-states-per-game 10 \
  --horizon-learner-turns 4 \
  --output runs/two-card-decision/samples.jsonl \
  --summary-output runs/two-card-decision/summary.json
```

Each sample stores:

```text
observation
selected_decision          # 0 = STOP_AFTER_ONE, 1 = PLAY_SECOND_HEURISTIC
stop_self_loss
stop_relative_loss
two_card_self_loss
two_card_relative_loss
decision_case
player_index
hand_count
seed
turn_index
```

The collector filters to turn-start states where the binary decision is
applicable, compares both choices with the same rollout oracle, and writes one
JSONL row per labeled state.

### Binary Decision Training And Evaluation

Train the compact binary classifier with:

```bash
python -m yellowstone.two_card_decision_training \
  runs/two-card-decision/samples.jsonl \
  --output-path models/yellowstone_two_card_decision.pt \
  --report-path runs/two-card-decision/training-report.json \
  --target relative_loss \
  --minimum-target-margin 3
```

`minimum-target-margin` removes rollout comparisons whose four-turn target
difference is too small to be a reliable hard label. At runtime, card and frame
choices still come from the precomputed heuristic one-card/two-card plans.

Evaluate the learned player against three heuristic players with:

```bash
python -m yellowstone.two_card_decision_evaluation \
  models/yellowstone_two_card_decision.pt \
  --games 100 \
  --threshold 0.5 \
  --confidence-margin 0.3 \
  --json-output runs/two-card-decision/evaluation.json
```

With confidence margin `0.3`, predictions between `0.2` and `0.8` fall back to
the standard heuristic. Only higher-confidence predictions override the number
of cards played.

### Binary Decision Run 001

Run 001 collected 4,881 horizon-4 continuing-game samples. The labels were
generated from `relative_loss`; 2,089 samples remained after requiring target
margin `>= 3`. The margin-3 model reached validation accuracy `0.5947` and
balanced accuracy `0.5673`. A policy that always trusts this classifier was
worse than heuristic-only play.

Confidence gating recovered most of the regression but has not demonstrated a
repeatable improvement:

```text
seed 10000, 100 games:
  learned p0 loss share    = 0.2660
  heuristic p0 loss share = 0.2818

seed 20000, 300 games:
  learned p0 loss share    = 0.2575
  heuristic p0 loss share = 0.2514

combined 400-game weighted average:
  learned p0 loss share    ~= 0.2596
  heuristic p0 loss share ~= 0.2590
```

The independent 300-game result did not reproduce the initial improvement, so
this model should be treated as parity at best, not as a heuristic replacement.
The gated policy overrode about one quarter of applicable decisions and used
the standard heuristic for the rest.

The shared-plan optimization reduced the 20-state collector smoke from about
11 seconds to 5.6 seconds. Learned evaluation is still expensive: about 2.1
seconds per game versus 0.05 seconds for heuristic-only, primarily because the
two-card heuristic enumerates all legal two-placement plans.

Next work should focus on label predictability and plan-enumeration cost before
collecting substantially more data. More samples alone improved validation only
slightly, from roughly chance to balanced accuracy `0.57`.

### Binary Decision Learning Curve

The collector now stores the actual source-game seed and supports
`--max-source-states-per-game`. This is important in continuing-game mode:
without a per-game limit, 19,499 samples came from only 27 game seeds. The
diverse run limited each game to 10 source states and collected:

```text
train:      19,633 samples / 2,002 seeds
validation:  4,909 samples /   501 seeds
```

The seed ranges were disjoint. Training used `relative_loss`, target margin 3,
50 epochs, class balancing, and the same fixed validation dataset for every
training size. The learning curve was:

```text
source states  eligible train  balanced accuracy
2,000          680             0.5403
5,000          1,737           0.5816
10,000         3,509           0.5766
19,633         6,844           0.5748
```

Accuracy peaked at 5,000 source states rather than continuing to improve. In
the high-confidence region for confidence margin 0.4, that model had rollout
label accuracy `0.673` versus heuristic accuracy `0.609`, with coverage `0.148`.
The model changed the heuristic card count on `0.058` of eligible validation
states.

However, the improvement in rollout-label accuracy did not transfer to match
quality. On unused seeds starting at 500000:

```text
5k model, confidence margin 0.3, 100 games:
  model p0 loss share     = 0.2749
  heuristic p0 loss share = 0.2414

5k model, confidence margin 0.4, 100 games:
  model p0 loss share     = 0.2453
  heuristic p0 loss share = 0.2414
```

The stricter gate returned performance close to the baseline but did not beat
it. Collecting more examples of the current 66-feature observation is therefore
not the next priority. The next experiment should add direct one-card/two-card
immediate negative-card deltas or revise the four-turn target before another
large collection run.

### Binary Advantage Regression

The same diverse-seed data was reused without discarding low-margin samples.
`yellowstone.two_card_advantage_training` predicts the continuous target:

```text
advantage = stop_relative_loss - two_card_relative_loss
```

Positive values favor the two-card plan. Targets are standardized from the
training split and learned with Huber loss. At runtime,
`yellowstone.two_card_advantage_evaluation` uses the model only when the
absolute predicted advantage exceeds a configured threshold; otherwise it
falls back to the standard heuristic.

```bash
python -m yellowstone.two_card_advantage_training \
  runs/two-card-decision-run-003-diverse-seeds/train-s20000.jsonl \
  --validation-dataset-path \
    runs/two-card-decision-run-003-diverse-seeds/validation-s5000.jsonl \
  --train-sample-limit 10000 \
  --output-path models/yellowstone_two_card_advantage_h4_s10000.pt \
  --report-path runs/two-card-advantage-run-001/training-s10000.json
```

Regression used every sample. Validation metrics improved with data size:

```text
train samples  MAE    RMSE   balanced sign accuracy
2,000          3.206  4.438  0.5357
5,000          3.080  4.317  0.5393
10,000         3.007  4.282  0.5446
19,633         2.902  4.175  0.5315
```

The 10k model had the best sign accuracy. On validation data, threshold `2.0`
changed the heuristic card count on `10.7%` of states, was correct on `59.7%`
of those changes, and had positive mean four-turn target improvement. That
apparent improvement did not transfer to matches on unused seeds starting at
600000:

```text
threshold 2.0, 100 games:
  model p0 loss share     = 0.2322
  heuristic p0 loss share = 0.2157

threshold 3.0, 100 games:
  model p0 loss share     = 0.2330
  heuristic p0 loss share = 0.2157
```

Advantage regression made better use of all samples and improved magnitude
prediction as data increased, but the current single-rollout four-turn target
still did not align with full-match quality. The next target-quality experiment
should average multiple common-random-number rollouts per state. Adding direct
one-card/two-card immediate negative-card deltas remains the lower-cost feature
experiment.

### Immediate Negative-Delta Features

The observation now includes two additional values after the heuristic bonus
features:

```text
one_card_negative_delta
two_card_negative_delta
```

They come from the same precomputed heuristic plans used for execution, so
observation construction does not simulate either plan again. Observation size
increased from 66 to 68; older datasets and models are intentionally
incompatible.

### Sorted-Hand And Played-Rank Features

Hands are now sorted by `rank_index, color` when cards are dealt or refilled.
Observation construction keeps that state order and does not sort on every
encoding call. This preserves a stable hand-slot representation without paying
the sorting cost repeatedly during dataset collection and evaluation.

The binary two-card observation also includes the ranks selected by the
heuristic one-card and two-card plans:

```text
one_card_rank
two_card_low_rank
two_card_high_rank
```

The two-card ranks are stored in ascending order, independent of the actual
play order. Observation size increased from 68 to 71; older two-card decision
and advantage datasets/models are intentionally incompatible.

Run 001 with the sorted-hand/rank observation collected new K=4 continuing
datasets:

```text
train:      4,919 samples / 501 source seeds
validation: 1,231 samples / 125 source seeds
```

The all-case advantage model stopped at epoch 8:

```text
validation MAE                    = 2.689
validation RMSE                   = 3.888
validation balanced sign accuracy = 0.5322
heuristic sign accuracy            = 0.5256
```

Short continuing evaluation on 100 unused seeds starting at 2500000 did not
show an improvement in `one_to_two_only` mode:

```text
threshold  p0 share  heuristic  paired delta  95% CI                 overrides
0.35       0.262205  0.252590   +0.009616     [-0.005680, +0.024911]  919
0.50       0.257870  0.252590   +0.005280     [-0.006919, +0.017480]  579
0.75       0.255908  0.252590   +0.003319     [-0.007191, +0.013829]  170
```

A one-to-two filtered model trained only on
`heuristic_stops_with_second_available` samples had weak balanced sign accuracy
despite higher raw sign accuracy:

```text
train samples                     = 2,140
validation samples                = 532
best epoch                        = 14
validation MAE                    = 3.343
validation RMSE                   = 4.601
validation sign accuracy          = 0.6184
validation balanced sign accuracy = 0.5292
heuristic sign accuracy            = 0.6090
```

It changed very few turns. Threshold `0.0` was slightly favorable on one 100
seed batch but reversed on the next:

```text
seed start  threshold  p0 share  heuristic  paired delta  95% CI                 overrides
2500000     0.00       0.248833  0.252590   -0.003757     [-0.014282, +0.006769]  44
2600000     0.00       0.250580  0.249199   +0.001381     [-0.011361, +0.014122]  49
```

The sorted-hand/rank features are mechanically working, but this first small
K=4 run is not a replacement for the previous original K=4
`one_to_two_only / threshold=0.5` candidate. The next useful step is either to
scale the new dataset to match the old 9,380-sample run before judging the
feature, or to use the rank features for bucket diagnostics/gates instead of
expecting the small supervised model to learn the full interaction.

After checking the sorted-hand implementation, the continuing-game deck
exhaustion refill path was found to bypass hand sorting. That path is used by
the K=4 continuing data collector, so run 001 can include inconsistent hand
orders after continuing refills. The refill path now applies the same
`sort_hand` helper as normal game refill.

Run 002 regenerated the same-size datasets with the corrected continuing
refill:

```text
train:      4,919 samples / 501 source seeds
validation: 1,231 samples / 125 source seeds
```

The all-case model remained weak:

```text
best epoch                        = 11
validation MAE                    = 2.695
validation RMSE                   = 3.882
validation sign accuracy          = 0.5508
validation balanced sign accuracy = 0.5261
heuristic sign accuracy            = 0.5256
```

The one-to-two filtered model improved versus run 001 and now separates the
validation target better than the heuristic sign baseline:

```text
train samples                     = 2,140
validation samples                = 532
best epoch                        = 14
validation MAE                    = 3.327
validation RMSE                   = 4.583
validation sign accuracy          = 0.6391
validation balanced sign accuracy = 0.5720
heuristic sign accuracy            = 0.6090
```

This does not yet establish match-quality improvement, but it means the
sorted-hand/rank feature path should not be dismissed based on run 001. The
next check should evaluate the run 002 one-to-two model in continuing play,
starting with threshold `0.0` and then small positive thresholds if override
coverage is high enough.

Run 001 collected 9,838 train samples from 1,001 seeds and 2,465 validation
samples from a separate 250 seeds. The advantage model produced validation MAE
`2.984`, RMSE `4.241`, and balanced sign accuracy `0.548`.

At advantage threshold `2.5`, validation changed the heuristic card count on
`3.25%` of states and was correct on `62.5%` of those changes. Match evaluation
on unused seeds showed a small improvement in both independent batches:

```text
seed 900000, 100 games:
  model p0 loss share     = 0.2438
  heuristic p0 loss share = 0.2533

seed 910000, 300 games:
  model p0 loss share     = 0.2509
  heuristic p0 loss share = 0.2538

combined 400-game weighted average:
  model p0 loss share     ~= 0.2492
  heuristic p0 loss share ~= 0.2537
```

The effect is small, but unlike the earlier classifier and regression runs, the
direction reproduced on the independent 300-game set. Keep the feature and
treat the model as promising rather than conclusively stronger. A larger
evaluation or a multi-rollout target is still needed before replacing the
heuristic policy.

### Plan Search Optimization And Larger Evaluation

The two-card plan search now evaluates second-placement effects analytically
and applies a full immutable state transition only to the selected second
action. The candidate ordering, bonus score, and negative-card delta are
unchanged and are checked against the previous exhaustive-transition method.
On a one-game CLI timing check, total wall time including model startup and the
heuristic baseline decreased from `11.82` seconds to `3.64` seconds. A later
1,000-game model evaluation averaged `0.566` seconds per learned-policy game;
the earlier 300-game run had anomalously averaged about `19.7` seconds.

The larger unused-seed evaluation starting at 930000 did not reproduce the
earlier improvement:

```text
1,000 games, threshold 2.5:
  model p0 loss share     = 0.262269
  heuristic p0 loss share = 0.262174
  model p0 win rate       = 0.234
  heuristic p0 win rate   = 0.226
```

The loss-share difference was `+0.000095`, effectively parity. The immediate
negative-delta features remain useful inputs, but the K=1 checkpoint is not
established as stronger than the heuristic.

### Multiple-Rollout Average Target

`choose_two_card_decision_by_rollout` and the dataset CLI now accept
`rollout_count`. Each one-card/two-card pair uses the same deterministic seed
set (common random numbers), and its self/relative loss targets are averaged
over all rollouts. `rollout_count=1` preserves the previous behavior.

The first K=4 run used 4,716 training samples from 502 source seeds and 1,171
validation samples from a separate 126 seeds. Fifty epochs overfit. A fixed
epoch sweep selected 10 epochs:

```text
epochs  validation MAE  RMSE   balanced sign accuracy
10      2.729           3.963  0.5474
20      2.803           3.991  0.5400
30      2.880           4.051  0.5310
50      2.990           4.180  0.5223
```

At threshold `1.0`, the 10-epoch model changed the heuristic card count on
`1.62%` of validation states, was correct on `68.4%` of those changes, and had
mean target improvement `+0.00496` per state. On unused seeds starting at
970000:

```text
500 games, threshold 1.0:
  model p0 loss share     = 0.258337
  heuristic p0 loss share = 0.258825
  model p0 win rate       = 0.234
  heuristic p0 win rate   = 0.236
```

This `-0.000488` loss-share difference is favorable but too small to establish
an improvement. K=4 reduced validation error when training stopped early, so
the next experiment should increase K=4 source diversity and select the epoch
from validation rather than using a fixed 50 epochs.

### K=4 Data Scaling And Early Stopping

Advantage training now supports multiple input JSONL files through repeated
`--additional-dataset-path` arguments. It also supports validation-loss early
stopping with `--early-stopping-patience` and
`--early-stopping-min-delta`. The best model weights are restored before final
metrics and model serialization; reports include `best_epoch` and
`epochs_trained`.

An additional 4,664 K=4 samples from 502 new source seeds increased training
data to 9,380 samples. With maximum 100 epochs and patience 5, training stopped
after 19 epochs and restored epoch 14:

```text
validation MAE                    = 2.680
validation RMSE                   = 3.860
validation balanced sign accuracy = 0.5722
heuristic sign accuracy            = 0.5423
```

This is a clear improvement in predicting the averaged four-turn target. The
best validation policy thresholds were `0.5` by mean target improvement and
`0.75` for a more conservative override rate:

```text
threshold  changed rate  changed accuracy  mean target improvement
0.50       10.85%        52.76%            +0.0587
0.75        6.32%        56.76%            +0.0539
```

However, neither transferred to p0 loss share on the same 500 unused seeds
starting at 990000:

```text
threshold  model loss share  heuristic loss share  model win  heuristic win
0.50       0.248996          0.239133              25.2%      28.2%
0.75       0.240609          0.239133              30.6%      28.2%
```

K=4 and more data improved target prediction, but the four-turn relative-loss
target remains insufficiently aligned with full-match loss share. More samples
of the same target are now lower priority than examining changed states and
identifying missing post-refill or hand-quality value within the required
four-turn horizon.

### Continuing-Game Evaluation

The two-card advantage evaluator now supports
`--continuing-learner-turns`. It uses the existing continuing-game rule: deck
exhaustion still settles negative cards and reshuffles them, but reaching 35
loss points does not end the match. Metrics use loss deltas from the initial
scores after a fixed number of player-0 turn starts.

The 9,380-sample K=4 early-stopped model was evaluated at threshold `0.75` on
100 unused seeds starting at 1010000 for 200 learner turns:

```text
                                  model       heuristic-only
p0 average added loss             192.48      185.03
p0 average added-loss share       0.254968    0.255515
average actions per match         1927.05     1895.81
```

The model's added-loss share was lower by `0.000547`, while its raw added loss
was higher because the model matches produced more total loss and actions.
Loss share is the primary continuing-game metric. This result is effectively
parity with a slight favorable direction, not evidence of a reliable win over
the heuristic.

### Direction-Restricted Continuing Policies

The evaluator supports two direction-restricted modes. In both modes, turns
outside the permitted override direction use the complete standard heuristic
action, so the model changes only the number of cards played:

- `one_to_two_only`: heuristic two-card turns stay heuristic; the model may
  only turn a heuristic one-card turn into the learned two-card plan.
- `two_to_one_only`: heuristic one-card turns stay heuristic; the model may
  only turn a heuristic two-card turn into the learned one-card plan.

Using the 9,380-sample K=4 model, threshold `0.75`, 200 learner turns, and the
same 50 unused seeds starting at 1030000 produced:

```text
policy             p0 added-loss share  delta vs heuristic  overrides
heuristic-only     0.278928             -                   -
one_to_two_only    0.250667             -0.028261           180
two_to_one_only    0.255837             -0.023091           932
```

Both restricted directions were favorable on this seed batch. The one-to-two
mode had the better loss share despite making far fewer changes. With only 50
continuing sequences, the size and direction ranking are preliminary and
should be reproduced on a larger independent batch before selecting a policy.

Larger continuing batches changed that interpretation. With threshold `0.75`
and 200 learner turns on 200 unused seeds starting at 1100000:

```text
policy             p0 added-loss share  heuristic  paired delta  95% CI
one_to_two_only    0.247363             0.249279   -0.001916     [-0.012248, +0.008415]
two_to_one_only    0.268447             0.249279   +0.019168     [+0.006428, +0.031907]
```

The two-to-one direction is harmful on the larger batch and should be dropped.
The one-to-two direction remains mildly favorable but noisy.

For one-to-two-only, threshold `0.25` was the best diagnostic setting on 100
seeds starting at 1200000:

```text
threshold  p0 added-loss share  heuristic  paired delta  95% CI
0.25       0.244867             0.266228   -0.021361     [-0.038724, -0.003997]
0.50       0.258226             0.266228   -0.008002     [-0.027597, +0.011593]
0.75       0.250896             0.266228   -0.015332     [-0.033915, +0.003251]
1.00       0.251009             0.266228   -0.015219     [-0.030123, -0.000315]
```

Unconditionally converting every heuristic one-card stop into a two-card turn
was strongly harmful (`0.460917` vs `0.266228`), so the learned gate is doing
real filtering work.

Independent 300-seed confirmation runs for the original K=4 model with
one-to-two-only threshold `0.25` showed consistent but small favorable deltas:

```text
seed start  p0 added-loss share  heuristic  paired delta  95% CI
1300000     0.240981             0.247964   -0.006983     [-0.018109, +0.004143]
1400000     0.249745             0.253219   -0.003474     [-0.013796, +0.006847]
```

### One-To-Two Filtered Advantage Model

Training can now filter samples by decision case with
`--decision-case-filter`. A one-to-two-only regression model was trained from
the K=4 datasets using only
`heuristic_stops_with_second_available` samples:

```text
train samples       = 4164
validation samples  = 521
best epoch          = 15
validation MAE      = 3.399
validation RMSE     = 4.521
validation sign acc = 0.6411
heuristic sign acc  = 0.6276
```

This improved validation sign accuracy within the one-to-two case, but did not
transfer robustly to continuing play. At threshold `0.25`:

```text
model                         seeds     p0 share  heuristic  paired delta  95% CI
filtered one-to-two           1500000   0.238227  0.254290   -0.016063     [-0.032713, +0.000586]
original K=4 one-to-two       1500000   0.243882  0.254290   -0.010408     [-0.029151, +0.008335]
filtered one-to-two           1600000   0.258165  0.251892   +0.006274     [-0.003818, +0.016365]
filtered + original confirm   1600000   0.245558  0.251892   -0.006334     [-0.016722, +0.004054]
```

The filtered model alone overfits the one-to-two validation target enough to
collapse on an independent 300-seed batch. Requiring both the filtered model
and original model to agree on a positive advantage reduced overrides
(`3077` to `1758` on the 1600000 batch) and restored the favorable direction,
but the effect size is still small and not statistically decisive.

The next useful experiment is not more of the same four-turn target. It should
diagnose the changed one-to-two states that hurt in full continuing play and
add a target or feature for post-refill hand quality / future forced damage,
because the current validation improvement does not reliably predict
continuing-game loss share.

### One-To-Two Threshold Sweep

The follow-up threshold search removed any target override-count constraint and
optimized only continuing-game player-0 loss share. The original K=4
early-stopped model was evaluated in `one_to_two_only` mode.

On 100 seeds starting at 1800000, lower and middle thresholds were better than
high thresholds:

```text
threshold  p0 share  heuristic  paired delta  overrides
0.00       0.247974  0.254162   -0.006188     1481
0.15       0.251904  0.254162   -0.002259     1231
0.25       0.249723  0.254162   -0.004440     1049
0.35       0.249906  0.254162   -0.004256      913
0.50       0.250069  0.254162   -0.004093      698
0.75       0.260819  0.254162   +0.006657      399
1.00       0.264067  0.254162   +0.009904      202
1.25       0.255372  0.254162   +0.001209       86
```

The apparent low-threshold advantage did not hold in larger batches. On 300
seeds starting at 1900000, thresholds `0.0` and `0.25` worsened, while `0.5`
improved:

```text
threshold  p0 share  heuristic  paired delta  95% CI
0.00       0.254974  0.247971   +0.007003     [-0.004738, +0.018744]
0.25       0.251365  0.247971   +0.003394     [-0.007743, +0.014530]
0.35       0.247093  0.247971   -0.000878     [-0.012037, +0.010281]
0.50       0.243494  0.247971   -0.004477     [-0.016246, +0.007292]
```

Another 300-seed batch starting at 2000000 favored `0.35`, but `0.5` was also
strong:

```text
threshold  p0 share  heuristic  paired delta  95% CI
0.25       0.246692  0.250608   -0.003916     [-0.013809, +0.005976]
0.35       0.238432  0.250608   -0.012176     [-0.022213, -0.002138]
0.50       0.240903  0.250608   -0.009705     [-0.019616, +0.000206]
```

A third 300-seed batch starting at 2100000 showed near-parity for `0.35` and a
clearer favorable direction for `0.5`:

```text
threshold  p0 share  heuristic  paired delta  95% CI
0.35       0.250286  0.251101   -0.000815     [-0.010862, +0.009233]
0.50       0.242827  0.251101   -0.008274     [-0.018204, +0.001656]
```

Combined over the new 300-seed confirmation batches with per-seed deltas:

```text
threshold  games  mean paired delta  95% CI                 overrides
0.25        600   -0.000261          [-0.007709, +0.007186]  6289
0.35        900   -0.004623          [-0.010647, +0.001401]  8082
0.50        900   -0.007485          [-0.013584, -0.001386]  6096
```

`0.5` is now the best threshold candidate for the original K=4 model in
`one_to_two_only` continuing play. It improves loss share more consistently
than `0.25` and avoids the high-threshold degradation seen at `0.75+`.
