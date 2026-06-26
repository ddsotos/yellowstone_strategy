import pytest

from yellowstone.two_card_decision_evaluation import (
    LearnedTwoCardDecisionBot,
    TwoCardDecisionEvaluationResult,
    evaluation_result_to_dict,
)
from yellowstone.evaluation import EvaluationSummary


def test_learned_binary_bot_keeps_runtime_decision_counts(tmp_path) -> None:
    # 学習botが二値判断回数を対戦評価用に保持する。
    bot = LearnedTwoCardDecisionBot(model_path=tmp_path / "model.pt", threshold=0.7)

    assert bot.pending_actions == ()
    assert bot.threshold == 0.7
    assert bot.confidence_margin == 0.0
    assert bot.decision_count == 0


def test_learned_binary_bot_rejects_invalid_confidence_margin(tmp_path) -> None:
    # confidence gateの幅が確率範囲を超える設定を拒否する。
    with pytest.raises(ValueError):
        LearnedTwoCardDecisionBot(
            model_path=tmp_path / "model.pt", confidence_margin=0.6
        )


def test_binary_evaluation_result_serializes_decision_rate() -> None:
    # 対戦結果に二枚出し判断率を含めてJSON化できる。
    summary = EvaluationSummary(
        match_count=1,
        win_rates=(1.0, 0.0, 0.0, 0.0),
        average_loss_scores=(1.0, 2.0, 3.0, 4.0),
        average_loss_shares=(0.1, 0.2, 0.3, 0.4),
        average_turn_count=10.0,
        total_elapsed_seconds=0.1,
        average_elapsed_seconds=0.1,
    )
    result = TwoCardDecisionEvaluationResult(
        model_vs_heuristic=summary,
        heuristic_only=summary,
        threshold=0.5,
        decision_count=10,
        two_card_decision_count=4,
    )

    assert evaluation_result_to_dict(result)["two_card_decision_rate"] == 0.4
    assert evaluation_result_to_dict(result)["heuristic_fallback_count"] == 0
