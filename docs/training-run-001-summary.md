# Training Run 001 Summary

この文書は、最初に実行した強化学習の小さな要約を残す。モデル本体、checkpoint、TensorBoardログ、評価JSON/CSVは生成物として `models/` と `runs/` に置き、Gitには含めない。

## Setup

- Command: `python -m yellowstone.training --total-timesteps 10000 --checkpoint-freq 2000`
- Device: CPU
- Algorithm: `MaskablePPO`
- Timesteps requested: `10000`
- Actual final timesteps logged: `10240`
- Evaluation frequency: `5000`
- Checkpoint frequency: `2000`

## Output Locations

- Final model: `models/yellowstone_maskable_ppo.zip`
- Best model: `models/best/best_model.zip`
- Checkpoints: `models/checkpoints/`
- Training logs and reports: `runs/training-run-001/`

These paths are intentionally ignored by Git.

## Evaluation

The saved model was evaluated with:

```bash
python -m yellowstone.model_evaluation models/yellowstone_maskable_ppo.zip --games 20
```

Results:

| Scenario | Player 0 win rate | Average loss scores | Average turns |
| --- | ---: | --- | ---: |
| model_vs_heuristic | 0.0 | `(5.0, 4.95, 4.85, 5.0)` | 16.40 |
| heuristic_only | 0.0 | `(13.35, 12.55, 8.4, 16.55)` | 44.95 |
| random_only | 0.1 | `(16.8, 16.2, 16.75, 16.0)` | 37.65 |

## Interpretation

This run proves that the learning pipeline can execute end to end: dependencies load, training runs, checkpoints are created, the final model is saved, and model evaluation runs.

It does not prove that the learned policy is strong. With only `10000` timesteps, player 0 did not win in the model-vs-heuristic evaluation. Next work should focus on longer runs, reward diagnostics, and checking whether short episode lengths indicate a reward or environment issue.
