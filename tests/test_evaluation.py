import pytest

from yellowstone.evaluation import (
    evaluate_policies,
    make_heuristic_policies,
    make_random_policies,
    run_match,
    summarize_results,
)


def test_run_match_returns_match_metrics() -> None:
    # 1試合を実行し、勝者・失点・ターン数・実行時間を返せることを確認する。
    result = run_match(make_heuristic_policies(), seed=1)

    assert len(result.loss_scores) == 4
    assert result.turn_count > 0
    assert result.action_count >= result.turn_count
    assert result.elapsed_seconds >= 0.0


def test_evaluate_policies_aggregates_multiple_seeds() -> None:
    # 複数seedの勝率、平均失点、平均ターン数、実行時間を集計できることを確認する。
    summary, results = evaluate_policies(
        make_heuristic_policies(),
        seeds=(1, 2),
    )

    assert len(results) == 2
    assert summary.match_count == 2
    assert len(summary.win_rates) == 4
    assert len(summary.average_loss_scores) == 4
    assert len(summary.average_loss_shares) == 4
    assert sum(summary.average_loss_shares) == pytest.approx(1.0)
    assert summary.average_turn_count > 0
    assert summary.total_elapsed_seconds >= summary.average_elapsed_seconds


def test_make_random_policies_can_run_match() -> None:
    # random botのpolicy群でも評価runnerを実行できることを確認する。
    result = run_match(make_random_policies(seed=1), seed=1)

    assert len(result.loss_scores) == 4
    assert result.action_count > 0


def test_summarize_results_handles_empty_results() -> None:
    # 結果が空でも0埋めのsummaryを返すことを確認する。
    summary = summarize_results((), player_count=4)

    assert summary.match_count == 0
    assert summary.win_rates == (0.0, 0.0, 0.0, 0.0)
    assert summary.average_loss_scores == (0.0, 0.0, 0.0, 0.0)
    assert summary.average_loss_shares == (0.0, 0.0, 0.0, 0.0)
    assert summary.average_turn_count == 0.0
