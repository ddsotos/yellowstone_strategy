# After-Refill State Value Learning

This note covers the state-value data target used by the current turn-level RL
reward.

The RL reward evaluates the learner's state immediately after the learner turn is
resolved and refill has happened. This differs from the older counterfactual
dataset target, which advanced NPC turns until the learner's next turn.

Use `--target-state after-refill` when collecting counterfactual value data for
the current reward shape:

```bash
python -m yellowstone.value_counterfactual_dataset \
  --source-games 300 \
  --source-state-limit 1000 \
  --actions-per-state 6 \
  --target-state after-refill \
  --exploratory-sources \
  --output runs/value-after-refill/samples.jsonl \
  --summary-output runs/value-after-refill/summary.json
```

Then train a value model from that dataset:

```bash
python -m yellowstone.value_training \
  runs/value-after-refill/samples.jsonl \
  --output-path models/yellowstone_state_value_after_refill.pt \
  --report-path runs/value-after-refill/training-report.json
```

The training report includes hand-count validation diagnostics:

```text
validation_loss_by_hand_count
validation_count_by_hand_count
```

Check these fields before increasing learned state-value reward weight. If hand
counts 4 or 5 have too few validation samples, the model may not reliably value
lower-hand-count positions even when the total validation loss looks acceptable.

In the first 2026-06-17 smoke run, `source-state-limit=1000` and
`actions-per-state=6` produced 5,582 samples. The hand-count histogram was:

```text
0: 0
1: 111
2: 188
3: 250
4: 400
5: 537
6: 4096
```

The resulting validation MSE was about `0.0118`, with hand-count 4 at about
`0.0119`, hand-count 5 at about `0.0152`, and hand-count 6 at about `0.0113`.
This is suitable for experiments, but still biased heavily toward six-card
states.
