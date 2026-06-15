# Observation And Action Space

この文書は、強化学習用の固定長観測表現と固定行動空間を定義する。

初期実装では NumPy や Gymnasium に依存せず、Python 標準の `tuple[int, ...]` と `tuple[bool, ...]` で扱う。必要になったら、この表現を wrapper 側で tensor へ変換する。

## Observation

実装は `src/yellowstone/observation.py` に置く。

入口:

```python
state_to_observation(state: GameState) -> tuple[int, ...]
observation_metadata() -> dict[str, int]
```

観測長は `OBSERVATION_SIZE` で固定する。2026-06-15 時点では `91`。

内訳:

1. 盤面: `2 + 3 * 4 + 3 * 3 = 23`
2. 現在プレイヤーの手札: `6 * 6 = 36`
3. プレイヤー概要: `5 * 4 = 20`
4. 現在プレイヤー one-hot: `5`
5. フェーズ one-hot: `3`
6. スカラー: `4`

### 盤面

盤面は、現在置かれているカードがすべて収まる3x3範囲に圧縮して表す。

先頭2値は以下:

```text
min_rank_index, min_x
```

`min_rank_index` は盤面にある中で一番小さい数字に対応する `y`、`min_x` は盤面にある中で一番左の列番号として扱う。この2値から、盤面カードが収まる3x3範囲を決める。盤面が空の場合はどちらも0にする。

続く12値は、3列それぞれの色パターン:

```text
col0_red, col0_blue, col0_green, col0_yellow,
col1_red, col1_blue, col1_green, col1_yellow,
col2_red, col2_blue, col2_green, col2_yellow
```

最後の9値は、3x3範囲内の各マスにあるカード総数。走査順は `y=min_rank_index..min_rank_index+2`、各行で `x=min_x..min_x+2`。

重ね置きされたカードは、列ごとの色枚数とセルごとの総枚数の両方に反映する。

### 手札

現在プレイヤーの手札だけを、最大6スロットで表す。

各スロットは以下の6値:

```text
present, red, blue, green, yellow, rank_index
```

空スロットはすべて0にする。

### プレイヤー概要

最大5人ぶんを固定長で持つ。

各プレイヤーは以下の4値:

```text
active, hand_count, negative_card_count, loss_score
```

存在しない5人目はすべて0にする。

### その他

- 現在プレイヤー: 最大5人の one-hot
- フェーズ: `play`, `refill`, `game_over` の one-hot
- スカラー: `cards_played_this_turn`, `deck_count`, `settlement_count`, `player_count`

## Turn-Level Action Space

現在の学習では、配置座標や3x3枠を直接学習対象にしない。
学習対象は「手札のどのスロットを使うか」だけを選び、実際の配置と補充は heuristic に従って決める。

実装は `src/yellowstone/turn_action_space.py` と `src/yellowstone/turn_env.py` に置く。

入口:

```python
turn_action_to_index(action: TurnAction) -> int
turn_action_from_index(index: int) -> TurnAction
resolve_turn_action(state: GameState, index: int) -> tuple[Action, ...]
legal_turn_action_indices(state: GameState) -> tuple[int, ...]
legal_turn_action_mask(state: GameState) -> tuple[bool, ...]
```

手札が6枚ある標準状態では、turn-level action は36通り。

1. 1枚プレイ: `6` 通り。手札スロット `0..5` のどれを1枚だけ使うか。
2. 2枚プレイ: `6 * 5 = 30` 通り。1枚目と2枚目に使う元の手札スロットの順序付きペア。

2枚プレイでは、1枚目を置いた後に手札indexが詰まる。
そのため2枚目は、手番開始時点の元スロットを基準に指定し、実行時に現在の手札indexへ変換する。

選ばれたカードの置き場所、3x3枠、補充元は、既存の heuristic 評価で決める。
これにより、初期学習では「何を出すか」に集中し、「どう置くか」はその場の失点を抑える既存ルールに寄せる。

turn-level の合法手 mask は、配置を実際に解決して検証しない。
ターン開始時の盤面は3x3枠内に収まっており、失点を許す限り手札カードは配置可能というルール前提に基づき、存在する手札スロットだけを `True` にする。
2枚プレイも、手番開始時点で存在する異なる2スロットの順序付きペアを合法とする。

## Low-Level Action Space

低レベル action space は、配置座標や3x3枠も含めて直接学習するための互換APIとして残す。

実装は `src/yellowstone/action_space.py` に置く。

入口:

```python
action_to_index(action: Action) -> int
action_from_index(index: int, state: GameState) -> Action
legal_action_indices(state: GameState) -> tuple[int, ...]
legal_action_mask(state: GameState) -> tuple[bool, ...]
action_space_metadata() -> dict[str, int]
```

行動空間の長さは `ACTION_SPACE_SIZE` で固定する。2026-06-13 時点では `1054`。

内訳:

1. 配置: `6 hand slots * 7 x positions * 25 frames = 1050`
2. ターン終了: `1`
3. 補充: `3`

### 配置 action

配置 action の index は以下で決める。

```text
hand_index, position.x, frame.y, frame.x
```

`position.y` は、現在状態の `hand[hand_index].rank_index` から復元する。これにより、rank と行の対応を保ったまま行動空間を小さくする。

### ターン終了 action

`EndTurnAction` は配置 action 群の直後の1 index に割り当てる。

### 補充 action

補充 action は以下の順に割り当てる。

1. `deck`
2. `negative_cards`
3. `none`

### legal_action_mask

`legal_action_mask(state)` は `ACTION_SPACE_SIZE` と同じ長さの `tuple[bool, ...]` を返す。

`legal_actions(state)` に含まれる行動だけ `True` にし、それ以外は `False` にする。
