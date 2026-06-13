# yellowstone_strategy

Yellowstone Park の戦略調査用プロジェクト。

まずは Python で正確なゲームロジックとテキストベースの開発者用操作環境を作り、その後に自己対戦、探索、強化学習、戦略評価へ拡張する。

## Development

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
python -m yellowstone.cli
```
