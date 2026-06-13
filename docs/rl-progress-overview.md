# Reinforcement Learning Progress Overview

この文書は、heuristic bot の処理最適化を行ったあたりから、強化学習を始められる状態に近づけるために何をしたかを、機械学習初心者向けにまとめる。

## 1. 処理を軽くした

最初の heuristic bot は、良い手を探すために多くの候補を細かく調べていた。強化学習では大量のゲームを何度も回すため、1手ごとの処理が重いと学習に時間がかかりすぎる。

そこで、選択ルールを少し簡略化し、特に「失点なしで2枚置けるか」を探す処理を軽くした。目的は、人間のように完璧に考えるbotを作ることではなく、学習相手として十分に一貫したNPCを高速に動かすこと。

## 2. 対戦をまとめて評価できるようにした

1ゲームだけ見ると、運の影響が大きい。そこで、複数seedで random bot や heuristic bot を対戦させ、勝率、平均失点、平均ターン数、実行時間を集計できる runner を作った。

これにより、「このbotは強そうか」「処理速度は学習に耐えるか」を数字で見られるようになった。

## 3. 学習用の環境APIを作った

強化学習では、AIが以下の流れでゲームとやり取りする。

1. 現在の状態を見る
2. 行動を1つ選ぶ
3. ゲームを進める
4. rewardを受け取る
5. 次の状態を見る

この形に合わせて、`YellowstoneEnv` を作った。学習対象は player 0 で、残り3人は heuristic bot が自動で動く。これにより、AIは自分の手番だけを考えればよい。

## 4. 観測を固定長の数字にした

機械学習モデルは、Pythonのオブジェクトやカード名をそのまま理解できない。そのため、盤面、手札、マイナスカード枚数、失点、現在プレイヤー、フェーズなどを、固定長の数値列に変換した。

固定長にした理由は、ニューラルネットワークへ毎回同じ形の入力を渡すため。

## 5. 行動を番号にした

AIは「このカードをこの場所に置く」という複雑な行動を直接扱うより、`0` から始まる番号を選ぶ方が扱いやすい。

そこで、配置、ターン終了、補充を固定の action index に対応させた。全体の行動空間は固定だが、実際にはその時点で選べない行動も多い。

## 6. action mask を用意した

選べない行動をAIが選ぶと学習効率が悪くなる。そこで、「今選んでよい番号だけ True にする」action mask を作った。

このプロジェクトでは、通常のPPOではなく、action mask を扱える `MaskablePPO` を初期候補にした。

## 7. reward を定義した

AIは reward を増やすように学習する。現在の baseline reward は以下を使う。

- 失点チャートが下がるとプラス
- マイナスカードが減ると小さくプラス
- ゲーム終了時に勝てばプラス、負ければマイナス

勝敗だけをrewardにすると、ゲーム終了までヒントが少なすぎるため、途中の良し悪しも少し入れている。

## 8. Gymnasium / Stable-Baselines3 につなげた

強化学習ライブラリで扱いやすくするため、既存の `YellowstoneEnv` を Gymnasium 形式に包む wrapper を作った。

学習ライブラリは以下を初期候補にしている。

- Gymnasium: 環境APIの標準形
- Stable-Baselines3: 学習アルゴリズム実装
- sb3-contrib MaskablePPO: action mask 対応のPPO

## 9. 観測を正規化した

ニューラルネットワークには、大きさがばらばらの数字をそのまま入れるより、0から1くらいの範囲に揃えた方が扱いやすい。

そのため、core APIでは整数の観測を維持しつつ、Gymnasium wrapperではデフォルトで `float32` の0..1正規化配列を返すようにした。

## 10. 学習と評価の入口を作った

最小学習スクリプトを追加し、seed、学習ステップ数、保存先、評価間隔を指定できるようにした。

さらに、学習済みモデルを heuristic bot / random bot と同じ指標で比較する評価スクリプトも追加した。

## 11. 保存・再開・結果出力を追加した

長時間学習では、途中で止めたり、後から比較したりできる必要がある。そのため、以下を追加した。

- checkpoint保存
- 保存済みモデルからのresume
- 学習設定のJSON report
- 評価結果のJSON/CSV出力

## 12. reward診断を追加した

reward が変な方向に学習を誘導していないか確認するため、heuristic / random の baseline reward を集計する診断を追加した。

これにより、学習前後で「rewardは高いが勝てない」といった問題を見つけやすくした。

## 現在できること

通常テスト:

```bash
pytest
```

reward診断:

```bash
python -m yellowstone.reward_diagnostics --policy heuristic --episodes 20 --json-output runs/reward-heuristic.json --csv-output runs/reward-heuristic.csv
```

RL依存導入後の学習:

```bash
python -m pip install -e ".[rl]"
python -m yellowstone.training --total-timesteps 10000 --checkpoint-freq 2000
```

学習済みモデル評価:

```bash
python -m yellowstone.model_evaluation models/yellowstone_maskable_ppo.zip --games 20 --json-output runs/model-eval.json --csv-output runs/model-eval.csv
```
