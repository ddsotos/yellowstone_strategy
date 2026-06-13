import pytest

from yellowstone.model_evaluation import (
    LearnedModelBot,
    evaluate_model,
    model_evaluation_dependencies_available,
)


def test_evaluate_model_reports_missing_optional_dependencies() -> None:
    # RL依存未導入環境では、モデル評価時に導入方法つきで失敗することを確認する。
    if model_evaluation_dependencies_available():
        pytest.skip("RL model evaluation dependencies are installed")

    with pytest.raises(ImportError, match=r'\.\[rl\]'):
        evaluate_model("missing-model.zip", seeds=(0,))


def test_evaluate_model_rejects_empty_seed_list() -> None:
    # 評価対象seedが空の場合は依存確認より前に明示的に失敗することを確認する。
    with pytest.raises(ValueError, match="seeds"):
        evaluate_model("missing-model.zip", seeds=())


def test_learned_model_bot_is_policy_adapter_type() -> None:
    # 学習済みモデルをBotPolicy互換で扱うadapterを作れることを確認する。
    bot = LearnedModelBot(model=object())

    assert bot.deterministic is True
