# State Value Experiment 2026-06-17

## Purpose

即時失点だけではなく、将来の失点割合を見込んだ状態評価を reward に入れると、学習が改善するかを確認した。

評価指標は、学習対象 player 0 の loss share とした。低いほど良い。

## Data

通常 heuristic と exploratory heuristic から、手番開始状態の教師データを作った。

```text
heuristic samples: 371,069
exploratory samples: 422,784
combined samples: 793,853
```

combined model の診断:

```text
validation_loss: 0.0159
constant_mean_mse: 0.0185
model_mse: 0.0158
correlation: 0.387
```

このモデルは、手札枚数が少ない状態ほど将来失点割合が高い、という相関を強く拾っていた。

その補正として、手札枚数別の平均を差し引く residual 方式を試した。また、実際の候補行動後の状態を増やすため、反事実データも作った。

```text
counterfactual samples: 21,318
counterfactual model validation_loss: 0.0131
counterfactual model correlation: 0.648
```

反事実モデルは、単純な on-policy 状態価値モデルより教師データ上の相関は良かった。

## RL Results

比較対象として、以前の良いモデルはおおよそ以下。

```text
previous best context:
model: yellowstone_turn_maskable_ppo_resume2381k_state_w02_two_bonus1_600k
p0 loss share: about 0.331
two-card rate: about 0.057
```

今回の主な結果:

| run | reward | eval games | p0 loss share | two-card rate |
| --- | --- | ---: | ---: | ---: |
| 015 | learned value w=1.0 | 100 | 0.341 | 0.034 |
| 016 | learned value w=1.0 + two-card 1.8 | 100 | 0.354 | 0.143 |
| 017 | residual learned value w=1.0 + two-card 1.8 | 100 | 0.338 | 0.134 |
| 018 | counterfactual value w=1.0 + two-card 1.8 | 100 | 0.337 | 0.126 |
| 018 checkpoint 3131344 | same as above | 200 | 0.335 | 0.129 |
| 019 continue 600k final | same as above | 200 | 0.358 | 0.170 |
| 019 checkpoint 3731344 | same as above | 200 | 0.351 | 0.167 |
| 020 | counterfactual value w=0.2 + two-card 1.8 | 200 | 0.345 | 0.137 |

## Findings

状態価値をそのまま reward に足すだけでは、今回の範囲では改善しなかった。

通常 heuristic 由来の状態価値は、低手札状態を悪く見すぎる傾向があった。そのため、2枚出しを避ける方向に働いた。

反事実データで value model の教師データ上の相関は改善した。しかし、RL reward に使った場合の loss share は、以前の最良モデルを超えなかった。

600k steps 継続しても悪化したため、同じ設定をさらに長く回す価値は低い。

value model を直接使って候補turn actionを選ぶ greedy policy も試した。正しくturn planを保持した実装では、100ゲームで p0 loss share は約0.372となり悪かった。これは、現在の state value model が単独で行動選択できるほど正確ではないことを示す。

## Next Candidates

次に試すなら、状態価値 `V(s)` ではなく、候補行動ごとの `Q(s, a)` または action preference を直接学習する方がよい。

理由:

- 現在の学習対象は36通りの turn action を選ぶ問題。
- `V(after)` の差分だけでは、候補行動の良し悪しを十分に分離できていない。
- 反事実ロールアウトは `action -> final loss share` の教師データを作れるため、`Q(s, a)` の方が自然。

現実的な次手:

1. 反事実データを `state, action_index, final_loss_share` として保存する。
2. 小さな supervised action-value model を学習する。
3. まず greedy action-value policy を評価する。
4. それが heuristic に近い水準まで行く場合だけ、RL の補助 reward または imitation pretraining に使う。

現時点では、state value reward を重くして長時間学習する方針は採用しない方がよい。
