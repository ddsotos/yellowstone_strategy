# CLI Guide

この文書は、開発者がテキストベースで Yellowstone Park の盤面と NPC の行動を確認するための CLI の使い方をまとめる。

現時点の CLI は、人間がカードを選んで遊ぶためのものではない。4人の heuristic NPC を先にゲーム終了までプレイさせ、その後で手番ログ、盤面、次に行動するプレイヤーの手札を Enter で順番に閲覧するデバッグ用ツールとして扱う。

## 実行方法

プロジェクトルートで以下を実行する。

```bash
python -m yellowstone.cli
```

開発用仮想環境を使う場合は以下を実行する。

```powershell
.venv\Scripts\python -m yellowstone.cli
```

## 基本仕様

- プレイヤー数は4人。
- 全プレイヤーは `HeuristicBot` で行動する。
- 初期 seed は `1`。
- 乱数は初期化と山札補充に使われる。
- 起動時に、まずゲーム終了まで全手番をシミュレーションしてログを作る。
- ログ生成後、Enter を押すたびに1手番分のログを表示する。
- 手番途中の盤面は表示せず、その手番の全行動と手番終了後の盤面だけを表示する。
- ゲーム終了までのログを読み終えると終了する。

CLI 引数はまだ用意していない。プレイヤー数や seed を変えて試したい場合は、現時点では `yellowstone.cli.run_npc_game(player_count=..., seed=...)` を Python から直接呼び出す。

例:

```powershell
.venv\Scripts\python -c "from yellowstone.cli import run_npc_game; run_npc_game(seed=2)"
```

## 画面の流れ

起動直後に、まずゲーム終了まで NPC 同士でプレイし、ログを生成する。

ログ生成後に以下を表示する。

1. 生成した手番数
2. 勝者、またはゲーム終了前に停止したこと
3. 初期状態の盤面
4. 初期状態のプレイヤー概要と手番プレイヤーの手札

その後、Enter を押すたびに以下を表示する。

1. その手番で選ばれた NPC 行動一覧
2. 手番終了後の盤面
3. 次に行動するプレイヤーの概要と手札

## 表示の読み方

### カード

カードは `R1` のように表示する。

- `R`: 赤
- `B`: 青
- `G`: 緑
- `Y`: 黄
- 数字: rank。内部の `rank_index` は `0..6` だが、表示では `1..7` にする。

### 盤面

盤面は7行で表示する。各セルはカードまたは `..` で表示する。

重ね置きされたセルは、一番上のカードと下にあるカード枚数を `B3+2` のように表示する。この場合、一番上は `B3`、その下に2枚ある。

### プレイヤー概要

プレイヤー概要は以下の形式で表示する。

```text
*P0: hand=6 negative=0 loss=5
 P1: hand=6 negative=0 loss=5
```

- `*`: 現在手番のプレイヤー
- `hand`: 手札枚数
- `negative`: マイナスカード置き場の枚数
- `loss`: 失点チャート位置

### 手札

現在手番プレイヤーの手札は以下の形式で表示する。

```text
P0 hand: 0:R1 1:B7 2:G3
```

左側の数字は `hand_index`。NPC が配置行動を選んだときの `hand[0]` などの表示と対応する。

### 行動

配置行動は以下の形式で表示する。

```text
Turn 1: P0
  1. place hand[2]=G3 at (3,2) frame=(2,0)
```

- `P0`: 行動したプレイヤー
- `hand[2]=G3`: 手札 index 2 の `G3` を出した
- `at (3,2)`: 盤面座標 `(x,y)` に置いた
- `frame=(2,0)`: 選んだ 3x3 枠の左上座標

ターン終了は以下の形式で表示する。

```text
Turn 2: P0
  1. place hand[0]=R1 at (0,0) frame=(0,0)
  2. end turn after one card
```

補充は以下の形式で表示する。

```text
Turn 3: P1
  1. place hand[4]=R2 at (1,1) frame=(1,1)
  2. place hand[4]=R3 at (1,2) frame=(1,1)
  3. refill source=deck
```

## 注意点

- この CLI は、heuristic bot の行動確認を目的にしている。
- 手動入力でカードや座標を選ぶ機能はまだない。
- 出力はデバッグ向けで、ユーザー向けUIとしては整えていない。
- 盤面や行動の妥当性は `src/yellowstone/game.py` の合法手生成と状態遷移に従う。
- 表示処理は `src/yellowstone/render.py` と `src/yellowstone/cli.py` に分かれている。
