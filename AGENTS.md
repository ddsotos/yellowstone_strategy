# Project Overview

ボードゲーム "Yellowstone Park" の戦略を調べるためのプロジェクト。

目的は、人間向けのオンライン版を作ることではなく、まず正確なゲームロジックを実装し、テキストベースで開発者が盤面確認や手動プレイをできる状態にすること。その後、自己対戦、探索、強化学習、戦略評価へ拡張する。

初期実装では Python を使う。TypeScript / React / Web UI は現時点では前提にしない。

# Build

想定コマンド:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
python -m yellowstone.cli
```

まだプロジェクト構成が存在しない場合は、上記を目標に最小構成から作る。

# Test

- テストフレームワークは `pytest` を使う。
- テストファイルは `tests/test_*.py` に配置する。
- 新機能やルール変更を追加するときは、対応するテストを同時に書く。
- スナップショットテストは避ける。
- 各テストの先頭に、日本語でテストの意図が分かる短いコメントを書く。
- ルール、状態遷移、合法手生成、決算、勝敗判定を変更した後は `pytest` を実行する。
- 乱数を使うテストでは seed を固定する。

# Conventions

- ゲームロジックは `src/yellowstone/` に置く。
- CLI 表示や入出力と、ルール判定・状態遷移を分離する。
- 中核 API は `GameState -> legal_actions -> apply_action -> GameState` の形を意識する。
- 盤面、手番、合法手、勝敗判定は型で明示する。
- `dataclasses`、`Enum`、型ヒントを使い、ゲーム状態を読みやすく保つ。
- 強化学習で扱いやすいよう、状態と行動は JSON 変換しやすい形にする。
- 乱数は関数内部で暗黙に生成せず、可能な限り seed または RNG を外から渡す。
- ゲームロジックは副作用の少ない純粋関数として実装する。
- テキスト盤面表示は `render.py`、開発者用手動プレイは `cli.py` に分ける。
- 最初からニューラルネットワーク実装に入らず、まず正しいゲーム環境、ランダム bot、簡単なヒューリスティック bot を作る。
- 変更は小さく、レビューしやすい単位にする。
- コミットコメントは日本語で書く。

# Suggested Structure

```text
src/yellowstone/
  __init__.py
  types.py        # Card, PlayerState, GameState, Action など
  game.py         # 合法手生成、状態遷移、決算、勝敗判定
  render.py       # テキスト盤面表示
  cli.py          # 開発者用CLIプレイ
  bots.py         # random bot / heuristic bot
  serialization.py

tests/
  test_*.py
```

# Design Priorities

1. ルールの正確さ
2. テストしやすさ
3. 強化学習環境としての扱いやすさ
4. テキストベースでのデバッグしやすさ
5. 実行速度

Web UI や見た目の作り込みは優先しない。

# TODO

- 作業前にこのTODOリストを確認する。
- 新しい未着手事項や後回しにした事項が出たら、このTODOリストに追加する。
- 完了した項目は削除するか、完了したことが分かるように更新する。
- TODOを変更した場合は、関連する設計書やテスト計画との整合も確認する。

## Current TODO

- [x] Python プロジェクトの最小構成を作る。
- [x] `docs/game-rules-design.md` をもとに、初期実装の対象ルールを整理する。
- [x] `src/yellowstone/types.py` に基本型を定義する。
- [x] `src/yellowstone/game.py` に初期化、合法手生成、状態遷移の最小実装を作る。
- [x] `src/yellowstone/render.py` にテキスト盤面表示を作る。
- [x] `pytest` の最小テストを追加する。
- [x] `docs/heuristic-bot-design.md` をもとに、`src/yellowstone/bots.py` に deterministic heuristic bot を実装する。
- [x] 強化学習前提の環境APIを設計する。`reset(seed)`, `step(action)`, `legal_action_mask`, `observation`, `reward`, `done`, `info` の形を決める。
- [x] 状態と行動のシリアライズを実装する。`GameState`、カード、盤面、合法手を JSON 変換しやすい形式へ変換できるようにする。
- [x] 観測表現を設計・実装する。盤面、手札、マイナスカード枚数、失点チャート、現在プレイヤー、フェーズを固定長の数値表現にする。
- [x] 行動空間を設計・実装する。配置、ターン終了、補充を固定 index に対応させ、合法手 mask と相互変換できるようにする。
- [x] 学習用 reward を設計する。即時報酬、失点チャート変化、マイナスカード増減、最終勝敗報酬の扱いを文書化する。
- [x] 学習対象1人対 heuristic bot 3人の環境を作る。学習対象プレイヤーの手番だけを外部 action にし、NPC手番は自動で進める。
- [x] ランダム bot と heuristic bot の対戦評価 runner を作る。複数 seed の勝率、平均失点、平均ターン数、実行時間を集計できるようにする。
- [x] 強化学習用の性能ベンチマークを追加する。ログ描画なしで多数ゲームを回し、1ゲームあたりの時間を測れるようにする。
- [x] 環境API、観測、行動mask、reward、NPC自動進行のテストを追加する。
- [x] 学習ライブラリを選定する。Gymnasium、Stable-Baselines3、sb3-contrib MaskablePPO を初期候補として採用し、依存関係と理由を文書化する。
- [x] Gymnasium wrapper を追加する。既存 `YellowstoneEnv` を Gymnasium の `reset` / `step` / `action_space` / `observation_space` 形式へ接続する。
- [x] action mask 対応を学習スクリプトへ接続する。`MaskablePPO` が使う `action_masks()` と評価時のmask扱いを検証する。
- [x] 観測の正規化方針を決める。現在の固定長整数観測を学習向け `np.ndarray` / tensor に変換し、スケール調整を検討する。
- [x] 最小学習スクリプトを追加する。seed、学習ステップ数、保存先、評価間隔を指定して実行できるようにする。
- [x] 学習済みモデル評価スクリプトを追加する。学習済みモデル、heuristic bot、random bot を同じ指標で比較する。
- [ ] モデル保存・再開・評価結果出力を整備する。checkpoint、resume、CSV/JSON出力を扱えるようにする。
- [ ] reward を実学習で検証する。学習ログと対戦ログを見て、変な行動を覚える場合は reward を調整する。

# Conversation Log

- ユーザーの重要なインプットと、Codexの端的な回答・判断を、時系列で専用Markdownに追記する。
- 記録先は `docs/conversation-log.md` とする。
- 仕様判断、設計判断、ルール修正、実装方針、検証結果など、後から文脈確認に使う内容を残す。
- 雑談、単純なpush依頼、同じ内容の繰り返し、長い実行ログは記録しない。
- 追記形式は「ユーザー入力」と「Codex回答」を1セットにし、短く要約して残す。
- 質問への回答は、実際の回答を残す。要約だけでは後で見て分からないため。
