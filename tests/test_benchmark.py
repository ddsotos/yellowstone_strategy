from yellowstone.benchmark import BenchmarkResult, benchmark_heuristic_games


def test_benchmark_result_computes_rates() -> None:
    # ベンチマーク結果から1ゲームあたり時間とgames/secを計算できることを確認する。
    result = BenchmarkResult(
        game_count=4,
        total_turn_count=40,
        total_action_count=100,
        elapsed_seconds=2.0,
    )

    assert result.seconds_per_game == 0.5
    assert result.games_per_second == 2.0


def test_benchmark_heuristic_games_runs_without_rendering() -> None:
    # 描画なしで複数ゲームの性能ベンチを実行できることを確認する。
    result = benchmark_heuristic_games(game_count=3, seed_start=1)

    assert result.game_count == 3
    assert result.total_turn_count > 0
    assert result.total_action_count >= result.total_turn_count
    assert result.elapsed_seconds >= 0.0
    assert result.seconds_per_game >= 0.0
