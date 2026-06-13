# Reward Validation

この文書は、現在の baseline reward を確認するための手順をまとめる。

## 目的

強化学習では、AIは reward が大きくなる行動を探す。reward の設計が悪いと、勝つためではなく、reward の穴を突くような行動を覚える可能性がある。

そのため、最初から長時間学習を回す前に、以下を確認する。

- heuristic learner の reward が極端に不自然でないか
- random learner と比べて heuristic learner の reward が概ね良いか
- 1ゲームごとの total reward、勝率、最終失点が後から確認できるか

## Baseline Diagnostics

RL依存を入れていない環境でも、以下で reward 診断を実行できる。

```bash
python -m yellowstone.reward_diagnostics --policy heuristic --episodes 20 --json-output runs/reward-heuristic.json --csv-output runs/reward-heuristic.csv
python -m yellowstone.reward_diagnostics --policy random --episodes 20 --json-output runs/reward-random.json --csv-output runs/reward-random.csv
```

見る項目:

- `average_total_reward`
- `win_rate`
- `average_learner_loss_score`
- episodeごとの `total_reward`

heuristic が random より明らかに悪い場合、reward が意図とずれている可能性がある。

## Training-Time Validation

RL依存を入れた後は、学習スクリプトで checkpoint と report を出す。

```bash
python -m yellowstone.training --total-timesteps 10000 --checkpoint-freq 2000 --report-path runs/train-report.json
```

学習後はモデル評価で heuristic / random baseline と比較する。

```bash
python -m yellowstone.model_evaluation models/yellowstone_maskable_ppo.zip --games 20 --json-output runs/model-eval.json --csv-output runs/model-eval.csv
```

学習済みモデルが random より悪い、または total reward は高いのに勝率や失点が悪い場合、reward を調整する。
