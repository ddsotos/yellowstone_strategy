import pytest

from yellowstone.action_value_evaluation import ContinuingEvaluationSummary
from yellowstone.two_card_advantage_evaluation import (
    ContinuingTwoCardAdvantageEvaluationResult,
    LearnedTwoCardAdvantageBot,
    continuing_evaluation_result_to_dict,
    evaluate_two_card_advantage_model_continuing_thresholds,
)


def test_advantage_bot_rejects_negative_threshold(tmp_path) -> None:
    # 回帰値の信頼閾値に負数を指定できない。
    with pytest.raises(ValueError):
        LearnedTwoCardAdvantageBot(
            model_path=tmp_path / "model.pt",
            advantage_threshold=-0.1,
        )


def test_advantage_bot_rejects_unknown_policy_mode(tmp_path) -> None:
    # 枚数変更方向は定義済みのモードだけを受け付ける。
    with pytest.raises(ValueError):
        LearnedTwoCardAdvantageBot(
            model_path=tmp_path / "model.pt",
            policy_mode="unknown",
        )


def test_advantage_bot_rejects_invalid_min_hand_count(tmp_path) -> None:
    # 1→2 gateの手札枚数条件は0..6だけを受け付ける。
    with pytest.raises(ValueError):
        LearnedTwoCardAdvantageBot(
            model_path=tmp_path / "model.pt",
            one_to_two_min_hand_count=7,
        )


def test_advantage_bot_rejects_invalid_exact_hand_count(tmp_path) -> None:
    # 1→2 gateの手札枚数完全一致条件も0..6だけを受け付ける。
    with pytest.raises(ValueError):
        LearnedTwoCardAdvantageBot(
            model_path=tmp_path / "model.pt",
            one_to_two_hand_count=-1,
        )


def test_advantage_bot_accepts_always_one_to_two_mode(tmp_path) -> None:
    # モデル選別なしの1枚→2枚baselineを評価モードとして指定できる。
    bot = LearnedTwoCardAdvantageBot(
        model_path=tmp_path / "model.pt",
        policy_mode="one_to_two_always",
    )

    assert bot.policy_mode == "one_to_two_always"


def test_continuing_advantage_result_serializes_loss_deltas() -> None:
    # 半無限評価の追加失点と失点割合をJSONへ保持する。
    summary = ContinuingEvaluationSummary(
        match_count=2,
        learner_turns=200,
        average_loss_deltas=(10.0, 11.0, 12.0, 13.0),
        average_loss_shares=(0.2, 0.24, 0.26, 0.3),
        average_action_count=1600.0,
        total_elapsed_seconds=2.0,
        average_elapsed_seconds=1.0,
    )
    result = ContinuingTwoCardAdvantageEvaluationResult(
        model_vs_heuristic=summary,
        heuristic_only=summary,
        advantage_threshold=0.75,
        decision_count=100,
        two_card_decision_count=20,
        heuristic_fallback_count=50,
        policy_mode="one_to_two_only",
        confirmation_model_path=None,
        confirmation_advantage_threshold=None,
        one_to_two_min_hand_count=6,
        one_to_two_hand_count=5,
        directional_override_count=10,
        paired_p0_loss_share_deltas=(-0.02, 0.0),
        paired_p0_loss_share_delta=-0.01,
        paired_p0_loss_share_ci95_low=-0.02,
        paired_p0_loss_share_ci95_high=0.0,
    )

    data = continuing_evaluation_result_to_dict(result)

    assert data["model_vs_heuristic"]["learner_turns"] == 200
    assert data["model_vs_heuristic"]["average_loss_deltas"] == (
        10.0,
        11.0,
        12.0,
        13.0,
    )
    assert data["policy_mode"] == "one_to_two_only"
    assert data["confirmation_model_path"] is None
    assert data["one_to_two_min_hand_count"] == 6
    assert data["one_to_two_hand_count"] == 5
    assert data["directional_override_count"] == 10
    assert data["paired_p0_loss_share_deltas"] == (-0.02, 0.0)
    assert data["paired_p0_loss_share_delta"] == -0.01


def test_multi_threshold_continuing_evaluation_rejects_empty_thresholds(tmp_path) -> None:
    # 半無限の複数閾値評価は、少なくとも1つの閾値を必要とする。
    with pytest.raises(ValueError):
        evaluate_two_card_advantage_model_continuing_thresholds(
            tmp_path / "model.pt",
            seeds=(1,),
            advantage_thresholds=(),
            learner_turns=1,
        )
