import pytest

from yellowstone.evaluation import EvaluationSummary
from yellowstone.model_evaluation import (
    LearnedModelBot,
    ModelEvaluationResult,
    evaluate_model,
    model_evaluation_result_to_dict,
    model_evaluation_dependencies_available,
    write_model_evaluation_result,
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


def test_model_evaluation_result_can_be_serialized(tmp_path) -> None:
    # 学習済みモデル評価をJSON/CSVの両方で出力できることを確認する。
    summary = EvaluationSummary(
        match_count=1,
        win_rates=(1.0, 0.0, 0.0, 0.0),
        average_loss_scores=(5.0, 6.0, 7.0, 8.0),
        average_loss_shares=(5 / 26, 6 / 26, 7 / 26, 8 / 26),
        average_turn_count=12.0,
        total_elapsed_seconds=0.5,
        average_elapsed_seconds=0.5,
    )
    result = ModelEvaluationResult(
        model_vs_heuristic=summary,
        heuristic_only=summary,
        random_only=summary,
    )
    json_output = tmp_path / "evaluation.json"
    csv_output = tmp_path / "evaluation.csv"

    write_model_evaluation_result(
        result,
        json_output=json_output,
        csv_output=csv_output,
    )

    assert model_evaluation_result_to_dict(result)["model_vs_heuristic"]["match_count"] == 1
    assert "model_vs_heuristic" in json_output.read_text(encoding="utf-8")
    assert "average_loss_shares" in json_output.read_text(encoding="utf-8")
    assert "scenario,match_count" in csv_output.read_text(encoding="utf-8")
