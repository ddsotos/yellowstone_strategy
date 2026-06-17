# Conversation Log

## 2026-06-09

### ユーザー入力

`agents.md` と `docs/game-rules-design.md` を置いたので、やりたいことが読み取れるか確認したい。

### Codex回答

Yellowstone Park のルールを実装し、最終的に戦略分析や強化学習へつなげたいプロジェクトだと判断した。初期方針としてはゲームロジック、合法手生成、状態遷移、勝敗判定を分離して実装する必要があると説明した。

### ユーザー入力

強化学習において TypeScript を使うのは正しい選択肢か質問した。

### Codex回答

TypeScript は Web UI や人間向けの検証環境には向くが、本格的な強化学習には Python + PyTorch/JAX などの方が現実的だと回答した。ただし、Web UI を作る前提なら TypeScript でゲームエンジンを作る選択も妥当だと説明した。

### ユーザー入力

実際に遊べる形は現状考えておらず、テキストベースで開発者がゲームできれば十分。この場合は Python がよいか質問した。

### Codex回答

その前提なら Python の方がよいと回答した。Python でゲームルール、合法手生成、状態遷移、テキスト盤面表示、CLI、random bot、自己対戦、強化学習まで一貫して扱う構成を推奨した。

### ユーザー入力

以上の内容を `AGENTS.md` に自由に反映してほしい。

### Codex回答

`AGENTS.md` を Python / pytest / CLI / 強化学習向けの方針に更新した。TypeScript、React、pnpm、Vitest 前提を外し、`src/yellowstone/` 配下にゲームロジック、CLI、render、bot、serialization を分ける構成を追記した。

### ユーザー入力

コミット、プッシュに関するルールを skill にまとめたい。コミットは TODO の単位で分け、プッシュ依頼があった場合はコミットも含めて実行してほしい。

### Codex回答

このリポジトリだけに適用するローカル skill として `.codex/skills/git-commit-push/` を追加した。TODO単位でのコミット分割、プッシュ依頼時の事前コミット、未関係のユーザー変更を含めないこと、履歴改変や force push を避けることを明文化した。

### ユーザー入力

TODO の一つ目である Python プロジェクトの最小構成作成に取り掛かってほしい。

### Codex回答

`pyproject.toml`、`README.md`、`src/yellowstone/__init__.py`、`src/yellowstone/cli.py`、`src/yellowstone/py.typed`、`tests/.gitkeep` を追加し、`python -m pip install -e ".[dev]"` と `python -m yellowstone.cli` を目標にした最小構成を作った。ゲームルール実装にはまだ入っていない。

### ユーザー入力

`pytest` のインストールを依頼した。

### Codex回答

通常のユーザー領域への `python -m pip install -e ".[dev]"` は権限エラーで失敗したため、プロジェクト内に `.venv` を作成し、`.venv\Scripts\python -m pip install -e ".[dev]"` で `pytest 9.0.3` をインストールした。`.venv` 上の CLI 実行は成功し、`pytest` は起動できるが、まだテストがないため `no tests ran` になっている。

### ユーザー入力

次の TODO に進むよう依頼した。

### Codex回答

`docs/game-rules-design.md` を初期実装向けに整理し、`docs/initial-implementation-scope.md` を追加した。4〜5人通常ルール、カード・山札、初期化、座標、3x3枠、配置、手番、補充、プラスポイント、決算、勝敗を初期対象にし、2〜3人ルール、上級ルール、Web UI、Undo、強化学習モデルは対象外にした。既存カードがあるマスへの配置時の扱いと、rank と y の対応を実装前確認事項として残した。

### ユーザー入力

既存カードがあるマスへの配置は重ね置き、既存カードがあるマスでは占有マス数は増えない、`rank 1 -> y=0` で進めてよいと回答した。強化学習で扱いやすければカード番号を0〜6にしてもよいと補足した。

### Codex回答

`docs/initial-implementation-scope.md` と `docs/game-rules-design.md` に反映した。内部表現は `rank_index=0..6`、表示・ルール説明上は `rank=rank_index+1` とし、`rank_index == y` で横列制限を判定する方針にした。既存カードがあるマスへの配置は重ね置きで、占有マス数は増えない設計として確定した。

### ユーザー入力

次の TODO に進むよう依頼した。

### Codex回答

`src/yellowstone/types.py` を追加し、`Color`、`Phase`、`RefillSource`、`Card`、`Position`、`Frame`、`PlayerState`、`GameState`、`PlaceCardAction`、`EndTurnAction`、`RefillAction` を定義した。カードは内部 `rank_index=0..6` を持ち、表示用 `rank` は `rank_index + 1` とした。重ね置きに対応するため、盤面は `dict[Position, tuple[Card, ...]]` とした。

### ユーザー入力

次の TODO に進むよう依頼した。

### Codex回答

`src/yellowstone/game.py` を追加し、デッキ生成、4〜5人初期化、3x3枠、配置合法性、合法手生成、カード配置、1枚終了、補充、山札切れ決算、勝敗判定の最小実装を行った。重ね置きは `Board = dict[Position, tuple[Card, ...]]` で扱い、既存マスへの配置では占有マス数を増やさない。ルール変更に対応して `tests/test_game.py` も追加した。

### ユーザー入力

次の TODO に進むよう依頼した。

### Codex回答

`src/yellowstone/render.py` を追加し、カード、重ね置きセル、7x7盤面、プレイヤー概要、ゲーム状態全体をテキスト表示できるようにした。重ね置きセルは一番上のカードと下にある枚数を `B3+2` のように表示する。`tests/test_render.py` で表示の主要要素を確認するテストを追加した。

### ユーザー入力

強化学習用の相手として、学習対象1人対 heuristic bot 3人を想定している。deterministic heuristic bot の行動ルール案について、重ね置きを優先すること、盤面上に多い色を出すこと、真ん中の数字から遠い数字を優先すること、枠内占有数優先を外すこと、真ん中から遠い数字の優先を重ね置きより重要にすることを指定した。これらの理由も文書化するよう依頼した。

### Codex回答

`docs/heuristic-bot-design.md` を追加し、18個の deterministic heuristic bot ルールとして整理した。重ね置き優先は他プレイヤーのチャート減少チャンスを減らすため、多い色を出すのは次の自分の手番までその色列が残らず後で出せない可能性があるため、真ん中から遠い数字を優先するのは端の行ほど後で置きにくくなる可能性が高いため、と理由を明記した。今後の TODO として `src/yellowstone/bots.py` への実装を追加した。

### ユーザー入力

heuristic bot が、ダメージなしで2枚置けるケースでも2枚目を置かないことがありそうだと指摘した。まずダメージなしで2枚置ける配置ペアを探し、ある場合はペアに含まれるカードの中でルール9〜13の優先度が高いものを選び、そのカードを含むペアのうちもう一枚もルール9〜13の優先度が高いものを選ぶよう変更したいと依頼した。ダメージなしで2枚置けない場合は従来通りでよいとした。

### Codex回答

`docs/heuristic-bot-design.md` を更新し、プレイフェーズで2枚ノーダメージ配置ペアを先に探索する方針に変更した。ペアがある場合は、単発優先ルールの高いカードを含むペアを優先し、その中で相方カードの単発優先度が高いペアを選ぶ。ペアがない場合は従来の単発評価と、プラスポイントがある場合の2枚目選択を使う。

### ユーザー入力

手札6枚で補充フェーズに入るのはバグなので、heuristic bot の「手札が6枚なら補充しない」を削除したいと指摘した。また、追加したルールと元のルールが混乱しているため整理できないか依頼した。

### Codex回答

`docs/heuristic-bot-design.md` を整理し直した。番号参照で混乱しないように「共通の配置評価」と「フェーズ別の行動ルール」を分離し、2枚ノーダメージ探索は共通評価を使ってペア内のカードを比較する形にした。補充フェーズでは、手札0枚かつマイナスカード6枚以上ならマイナスカードから補充し、それ以外は山札から補充する方針に変更し、手札6枚の分岐は削除した。

### ユーザー入力

現在の変更を push した後、行動ルールのインターフェースと行動ルールのテストを作成するよう依頼した。

### Codex回答

heuristic bot 設計文書の整理をコミットして push した。その後、`src/yellowstone/bots.py` に `BotPolicy`、`HeuristicBot`、`choose_heuristic_action`、共通配置評価の入口を追加し、`tests/test_bots.py` で補充選択、ゲーム終了時の無行動、真ん中から遠い数字優先、重ね置き優先、2枚ノーダメージ探索、1枚終了判断を確認するテストを追加した。

### ユーザー入力

行動ルールに従ったプレイを確認するため、4人NPCでゲーム終了までプレイさせ、最初は盤面と手番プレイヤーの手札を表示し、Enter を押すたびに行動、盤面、次に行動するプレイヤーの手札を表示するよう依頼した。

### Codex回答

`python -m yellowstone.cli` で4人 heuristic NPC のステップ実行を開始できるようにした。初期状態として盤面、プレイヤー概要、現在プレイヤーの手札を表示し、Enter ごとに選択行動を表示して `apply_action` 後の盤面と次に行動するプレイヤーの手札を表示する。表示補助関数のテストを `tests/test_cli.py` に追加した。

### ユーザー入力
CLI の説明書を用意してほしい。

### Codex回答
`docs/cli.md` を追加し、4人 heuristic NPC のステップ実行 CLI について、実行方法、表示内容、カード・盤面・行動ログの読み方、現時点の制限を説明した。`README.md` からも参照できるようにした。

### ユーザー入力
Enter を押すたびに1アクションではなく、そのプレイヤーの全プレイを進め、次プレイヤーの開始時にするよう依頼した。途中の盤面表示は不要と指定した。

### Codex回答
CLI の進行単位を1アクションから1手番に変更し、現在プレイヤーの行動をまとめて適用してから行動一覧と次プレイヤー開始時の盤面・手札を表示する形にした。`docs/cli.md` も手番単位の説明に更新した。

### ユーザー入力
全プレイを終えた後で、そのログを Enter を押しながら閲覧する形式に変更したい。

### Codex回答
CLI を、起動時に4人 heuristic NPC のゲームを最後まで先にシミュレーションして手番ログを作成し、その後 Enter ごとに1手番分のログを閲覧する形式に変更した。途中盤面は出さず、各ログにはその手番の行動一覧と手番後の盤面・次プレイヤーの手札を表示する。

### ユーザー入力
push を依頼したうえで、CLI の全プレイログ生成が重いため、2枚ノーダメージ探索で1枚だけなら失点しない候補を先に探してから2枚組を探すなどのリファクタリングを依頼した。

### Codex回答
CLI ログ閲覧機能を `NPC対戦ログ閲覧CLIを追加` としてコミット・push した。その後、heuristic bot の2枚ノーダメージ探索で、1枚目からマイナスカードが増える候補を先に除外してから2枚目候補を展開するようにした。さらに固定の3x3枠計算をキャッシュし、全ログ生成の実測を約26秒から約1.9秒に短縮した。

### ユーザー入力
選択ルールを変更してもよいので、機械学習を回せるレベルまで処理時間をさらに短縮したい。

### Codex回答
2枚ノーダメージの全ペア最適化と2枚目先読みをやめ、1枚目はノーダメージ候補の共通評価だけで決め、2枚目は実際に1枚目を置いた後の状態で改めて選ぶ貪欲ルールに変更した。さらに bot の候補評価では `GameState` を生成せず、盤面と枠からマイナスカード枚数・プラスポイントを直接計算するようにした。全ログ生成は単発起動込みで約0.14秒、同一プロセス内100ゲームでは約35ms/ゲームまで短縮した。

### ユーザー入力
push を依頼し、強化学習を行うまでに必要なことを TODO に書くよう依頼した。

### Codex回答
高速化作業をコミットした後、`AGENTS.md` の Current TODO に、強化学習前提の環境API、状態・行動シリアライズ、観測表現、固定行動空間、reward設計、学習対象1人対heuristic bot 3人の環境、評価runner、性能ベンチ、環境APIテストを追加した。

### ユーザー入力
TODOの上2つ、強化学習前提の環境API設計と、状態・行動シリアライズ実装を依頼した。

### Codex回答
`docs/rl-environment-design.md` を追加し、`reset(seed)`、`step(action_index)`、`observation`、`legal_action_mask`、`reward`、`done`、`info`、NPC自動進行、乱数管理の方針を整理した。`src/yellowstone/serialization.py` を追加し、`GameState`、盤面、カード、プレイヤー状態、各Action、合法手一覧をJSON互換dictへ変換・復元できるようにした。

### ユーザー入力
次の2つのTODO、観測表現の設計・実装と、行動空間の設計・実装を依頼した。

### Codex回答
`docs/observation-action-space.md`、`src/yellowstone/observation.py`、`src/yellowstone/action_space.py` を追加した。観測は盤面、現在手札、プレイヤー概要、現在プレイヤー、フェーズ、スカラーを含む固定長 `tuple[int, ...]` とし、行動空間は配置1050、ターン終了1、補充3の固定長1054 index と合法手 mask にした。

### ユーザー入力
pushしたうえで、次の2つのTODO、学習用reward設計と、学習対象1人対heuristic bot 3人の環境作成を依頼した。まずreward設計に対応した。

### Codex回答
前回分をTODO単位でコミットしてpushした後、`docs/reward-design.md` と `src/yellowstone/rewards.py` を追加し、失点チャート変化、マイナスカード増減、終端勝敗rewardを組み合わせたbaseline rewardを定義した。

### ユーザー入力
続けて、学習対象1人対heuristic bot 3人の環境作成に対応した。

### Codex回答
`src/yellowstone/env.py` を追加し、`YellowstoneEnv.reset(seed)` と `step(action_index)` で学習対象1人の行動だけを外部入力にし、他プレイヤーをheuristic botで自動進行する環境を実装した。戻り値は観測、reward、done、合法手mask、infoを含む。

### ユーザー入力
pushしたうえで、次の2つのTODO、対戦評価runnerと強化学習用性能ベンチマークの追加を依頼した。まず対戦評価runnerに対応した。

### Codex回答
`RandomBot` と `src/yellowstone/evaluation.py` を追加した。任意のBotPolicy群で1試合を実行する `run_match`、複数seedを集計する `evaluate_policies`、勝率、平均失点、平均ターン数、実行時間をまとめる `EvaluationSummary` を実装した。

### ユーザー入力
続けて、強化学習用の性能ベンチマーク追加に対応した。

### Codex回答
`src/yellowstone/benchmark.py` を追加し、heuristic bot同士のゲームをログ描画なしで複数回実行して、総ターン数、総アクション数、経過秒数、1ゲームあたり秒数、games/secを測れるようにした。

### ユーザー入力
pushしたうえで次の2つのTODOを依頼したが、未完了TODOは環境API、観測、行動mask、reward、NPC自動進行のテスト追加のみだった。

### Codex回答
既存の単体テストに加えて `tests/test_rl_integration.py` を追加し、`YellowstoneEnv.reset/step` を通した観測、合法手mask、reward、NPC自動進行、ゲーム終了後step拒否の統合確認を追加した。これにより強化学習前TODOはすべて完了扱いにした。

### ユーザー入力
実際に強化学習を回すまでに必要な8項目をTODOに追加し、そのうち1つ目の学習ライブラリ選定と2つ目のGymnasium wrapper追加を依頼した。

### Codex回答
`AGENTS.md` に実学習向けTODOを8項目追加し、学習ライブラリは Gymnasium、Stable-Baselines3、sb3-contrib MaskablePPO を初期候補にした。`docs/training-library-selection.md` に理由と依存関係を記録し、`src/yellowstone/gym_env.py` に `YellowstoneGymEnv` を追加して、既存 `YellowstoneEnv` を Gymnasium の `reset` / `step` / `action_space` / `observation_space` / `action_masks()` へ接続した。

### ユーザー入力
push後、TODOの未実施上4つを依頼した。

### Codex回答
前回分を `Gymnasium wrapperと学習ライブラリ選定を追加` としてpushした。続けて、MaskablePPO 用のaction mask接続、観測正規化方針、最小学習スクリプト、学習済みモデル評価スクリプトを追加した。観測は core API では整数tupleを維持し、Gymnasium wrapperではデフォルトで `float32` の0..1正規化配列にする方針にした。

### ユーザー入力
push後、残るTODOの実施と、処理最適化以降ここまでの概要を機械学習初心者にも分かるように別文書へまとめることを依頼した。

### Codex回答
前回分を3コミットに分けてpushした。残るTODOとして、学習時のcheckpoint保存、保存済みモデルからのresume、学習report、モデル評価のJSON/CSV出力、reward診断を追加した。概要文書として `docs/rl-progress-overview.md` を追加し、処理最適化から学習環境・観測・行動mask・reward・Gymnasium接続・学習/評価入口までを初心者向けに整理した。

### ユーザー入力
学習評価で正常終了しない試合があり、合法手なしはゲームルール上あり得ないのではないかと確認した。その後、`EndTurnAction` で手札0枚のままターン終了できる状況を改善し、必ず山札補充かマイナスカードからの回復を入れるよう依頼した。

### Codex回答
未終了試合は、1枚置いて `EndTurnAction` した結果、手札0枚のまま `Phase.PLAY` で次手番に入り、合法手0で止まっていたと確認した。`EndTurnAction` 適用時に現在プレイヤーの手札が空なら次プレイヤーへ進めず `Phase.REFILL` に入るよう変更し、山札補充または条件を満たす場合のマイナスカード補充を必須にした。20k checkpoint の再評価では `game_over 50/50` になり、合法手なし停止が解消した。

### ユーザー入力
一旦、学習対象は「何をプレイするか」に集中させ、「どうプレイするか」は heuristic と可能な限り同じ、その場の失点を抑える方針にしたいと依頼した。手札6枚では、2枚プレイの `6*5` 通りと1枚プレイの6通り、計36通りにまとめたいと指定した。

### Codex回答
turn-level action space を追加し、1枚プレイ6通り、2枚プレイ30通りの計36 action にした。選ばれた手札スロットの実際の配置、3x3枠、補充元は heuristic で解決する `YellowstoneTurnEnv` / `YellowstoneTurnGymEnv` を追加し、学習スクリプトとモデル評価adapterをturn-level actionに切り替えた。

### ユーザー入力
プレイ開始時は盤面のカードが3x3に収まっているため、失点を許すなら配置できないことは現実のゲームでは起きないと指摘し、その前提を反映するよう依頼した。

### Codex回答
turn-level mask 生成を、36通りすべての `resolve_turn_action` 実行から、手札スロットの存在だけを見る軽量方式へ変更した。1枚プレイは存在する手札スロット、2枚プレイは存在する異なる2スロットの順序付きペアを合法とし、配置不能はルール前提に反する実装バグとして実行時に検出する方針にした。

### ユーザー入力
5分学習で60k指定がタイムアウトした際、途中結果を使わず6144 timestepsで最初からやり直したのは意図と外れると指摘した。フィードバックを早く得るため、タイムアウト時点のcheckpointを利用する運用と、ここまでの作法を機械学習実行時のskillとしてまとめるよう依頼した。

### Codex回答
タイムアウト時はまずcheckpointやログを確認し、使える途中成果を評価すべきだったと判断した。`.codex/skills/yellowstone-ml-training/` を追加し、短時間学習ではcheckpointを早めに作る、タイムアウト時は再実行せず最新checkpointを評価する、平均失点はgame_over率や停止理由と一緒に解釈する、random比較は同条件でない限り断定しない、といった運用を明文化した。
### ユーザー入力
機械学習評価では、当面 `game_over` しているかどうかのチェックをスキップし、平均失点ではなく「全プレイヤー失点合計に占める自身の失点割合」で評価するよう依頼した。また、学習時にゲーム状況をどう入力しているか質問した。
### Codex回答
評価サマリに `average_loss_shares` を追加し、各試合で `player_loss / sum(all_player_losses)` を計算してから試合平均する方針にした。CLI表示も `avg_loss` ではなく `loss_share` を主表示に変更した。機械学習実行スキルにも、当面は `game_over` チェックを必須にせず、失点割合を主指標にする運用を反映した。

### ユーザー入力
盤面の学習入力を単純化し、盤面にある最小数字と一番左の列番号、3列の色パターン、3x3内の各マスのカード枚数に収められるはずだと指摘した。

### Codex回答
盤面観測を `7 * 7 * 4 = 196` から `2 + 3 * 4 + 3 * 3 = 23` に圧縮した。先頭に `min_rank_index` と `min_x`、次に3列それぞれの色別枚数、最後に3x3各マスのカード総数を入れる形式にし、全体の観測長は `91` になった。

### ユーザー入力
色ごとに固有の特徴はないため、赤・青・緑・黄の絶対色を入力せず、相対色one-hotのまま「絶対色だけ消す」よう依頼した。

### Codex回答
盤面から列ごとの絶対色枚数12値を削除し、盤面は `min_rank_index`, `min_x`, 3x3セルごとのカード総数9値だけにした。手札の色one-hotは、盤面3列の右列色、中央列色、左列色、未出色を相対IDへ割り当てる方式に変更した。絶対色を消しつつ、手札内で同じ色かどうかは保持し、全体の観測長は `79` になった。

### ユーザー入力
そのほかに圧縮できる要素として提示した1〜6、つまり現在プレイヤー、人数、フェーズ、手番内プレイ枚数、active、5人対応、山札枚数の細かい値の圧縮を適用するよう依頼した。

### Codex回答
turn-level学習では行動選択時点が現在プレイヤーのプレイフェーズ開始時に揃うため、`current_player`、`phase`、`cards_played_this_turn`、`player_count` を観測から削除した。4人学習前提としてプレイヤー概要を4人分にし、各プレイヤーの `active` を削除して `hand_count`, `negative_card_count`, `loss_score` の3値にした。山札枚数は `0`, `1..6`, `7..18`, `19+` の4段階へ丸め、観測長は `61` になった。
### ユーザー入力
10時間かけて、確認が必要になった作業は中断しつつ、強化学習の改善を試してほしいと依頼した。
### Codex回答
状態評価 reward を軽量化し、`YELLOWSTONE_STATE_VALUE_REWARD_WEIGHT` で調整可能にした。さらに `gamma` / `n_steps` を学習CLIで指定できるようにし、方策診断コマンド `yellowstone.policy_diagnostics` を追加した。480kモデルから状態評価 reward で継続し、p0 loss share は 0.446 付近から 0.363、さらに2枚ターンボーナスを加えた継続で最良 0.331 付近まで改善した。ただし heuristic 同士の p0 loss share 0.241 付近にはまだ届かず、2枚選択率も learned 5%前後に対して heuristic は47%前後で、主な残課題は2枚プレイ判断の学習不足と判断した。
### ユーザー入力
失点なし2枚出し reward は、2枚出し後に引ける枚数に比例させ、その報酬を5倍にできるか質問した。
### Codex回答
`YELLOWSTONE_NO_DAMAGE_TWO_CARD_TURN_REWARD_WEIGHT` を「最大ボーナス」として扱い、失点なし2枚ターンでは `最大ボーナス * 引ける枚数 / 6` を加える形に変更した。5倍設定は最大ボーナス `3.0` とし、手札6枚では `+1.0`、手札2枚では `+3.0` になる。強めの設定だが、失点なし2枚出し限定の実験スイッチとして試す価値があると判断した。

### ユーザー入力
どんな場合でも2枚出しには reward を付け、ただし3倍設定で30分程度試すよう依頼した。

### Codex回答
2枚出し reward を失点なし限定から、失点が増える2枚出しにも付く仕様へ変更した。新しい環境変数は `YELLOWSTONE_TWO_CARD_TURN_REWARD_WEIGHT` とし、既存の失点なし用環境変数は互換用に残した。3倍設定として最大ボーナス `1.8` を使い、手札6枚では `+0.6`、手札2枚では `+1.8` になる。150k steps の継続学習では2枚出し率は約15.7%まで上がったが、100ゲーム評価の p0 loss share は約0.350で悪化したため、この設定は2枚出しを増やす効果はあるが、現時点では失点割合改善にはつながっていないと判断した。
### ユーザー入力
状態価値を学習する準備を整えるよう依頼した。直前の議論では、heuristic-only だと手番開始時の手札6枚が多くなるため、失点なしで2枚出せる場合でも一定確率で1枚止めする探索用 heuristic で手札枚数分布をずらす案を採用した。
### Codex回答
評価用 heuristic は固定したまま、データ収集専用の `ExploratoryHeuristicBot` を追加した。手番開始状態の observation と、ゲーム終了時の該当プレイヤー失点割合を JSONL に保存する `yellowstone.value_dataset`、その教師データから小さな教師あり value model を学習する `yellowstone.value_training`、使い方をまとめた `docs/state-value-learning.md` を追加した。RL reward への接続は次段階とした。

### ユーザー入力
10時間の自律作業として、状態価値学習の検証と改善を依頼した。
### Codex回答
通常 heuristic / exploratory heuristic から状態価値データを作り、学習済み状態価値 reward を実装して複数設定でRL検証した。通常の状態価値は低手札状態を悪く見すぎて2枚出しを抑制し、反事実データで改善しても以前の最良 p0 loss share 約0.331を超えなかった。600k steps継続でも悪化したため、state value rewardを重くして長時間回す方針は採用しないと判断した。結果は `docs/state-value-experiment-2026-06-17.md` に整理し、次は `Q(s,a)` または action preference を直接学習する方針を候補にした。
