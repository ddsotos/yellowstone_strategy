from yellowstone.action_value_evaluation import (
    ActionValueBot,
    ActionValueEvaluationResult,
    action_value_evaluation_result_to_dict,
    write_action_value_evaluation_result,
)
from yellowstone.evaluation import EvaluationSummary


def test_action_value_bot_is_policy_adapter_type(tmp_path) -> None:
    # action-value modelをBotPolicy互換adapterとして扱える。
    bot = ActionValueBot(model_path=tmp_path / "model.pt")

    assert bot.pending_actions == ()
    assert bot.immediate_loss_penalty == 0.0


def test_action_value_evaluation_result_can_be_serialized(tmp_path) -> None:
    # action-value評価結果をJSON/CSVの両方で出力できる。
    summary = EvaluationSummary(
        match_count=1,
        win_rates=(1.0, 0.0, 0.0, 0.0),
        average_loss_scores=(5.0, 6.0, 7.0, 8.0),
        average_loss_shares=(5 / 26, 6 / 26, 7 / 26, 8 / 26),
        average_turn_count=12.0,
        total_elapsed_seconds=0.5,
        average_elapsed_seconds=0.5,
    )
    result = ActionValueEvaluationResult(
        model_vs_heuristic=summary,
        heuristic_only=summary,
        random_only=summary,
    )
    json_output = tmp_path / "evaluation.json"
    csv_output = tmp_path / "evaluation.csv"

    write_action_value_evaluation_result(
        result,
        json_output=json_output,
        csv_output=csv_output,
    )

    assert action_value_evaluation_result_to_dict(result)["model_vs_heuristic"][
        "match_count"
    ] == 1
    assert "model_vs_heuristic" in json_output.read_text(encoding="utf-8")
    assert "scenario,match_count" in csv_output.read_text(encoding="utf-8")
