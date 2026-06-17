# State Value Learning

状態価値は、手番開始時の状態からゲーム終了まで進めたとき、そのプレイヤーが最終的に負う失点割合を予測する補助モデルとして扱う。

主目的は、即時失点だけでは見えない「後で痛い状態」を reward に反映すること。

## Data

教師データは heuristic bot のロールアウトから作る。

- 対象状態: `Phase.PLAY` かつ `cards_played_this_turn == 0` の手番開始状態
- 入力: 既存の正規化済み observation
- ラベル: そのゲーム終了時の `player_loss / sum(all_player_losses)`
- 出力形式: JSON Lines

通常 heuristic のデータは強い基準行動の状態分布を表す。一方で、低い手札枚数の手番開始状態は少なくなりやすい。

そのため、データ収集用には `ExploratoryHeuristicBot` を別に用意する。この bot は、失点なしで2枚出せる場合でも、手札枚数に応じた確率で1枚だけ出して止める。評価用・NPC用の deterministic heuristic とは分けて扱う。

初期設定:

```text
hand=6: 35%
hand=5: 25%
hand=4: 15%
hand<=3: 0%
```

## Commands

通常 heuristic でデータを作る。

```bash
python -m yellowstone.value_dataset --games 1000 --output runs/value/value-samples.jsonl --summary-output runs/value/value-summary.json
```

探索 heuristic で手札枚数分布を広げる。

```bash
python -m yellowstone.value_dataset --games 1000 --exploratory --output runs/value/value-samples-exploratory.jsonl --summary-output runs/value/value-summary-exploratory.json
```

value model を学習する。

```bash
python -m yellowstone.value_training runs/value/value-samples.jsonl --output-path models/yellowstone_state_value.pt --report-path runs/value/value-training-report.json
```

`runs/` と `models/` は生成物なので Git には入れない。

## Next Step

次の実装では、学習済み value model を読み込み、RL reward に以下の形で接続する。

```text
reward = immediate_reward + lambda * (V(after) - V(before))
```

これにより、1枚出しで一時的に失点を避けても、将来の失点割合が悪くなる状態なら低く評価できる。
