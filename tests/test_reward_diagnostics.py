from yellowstone.reward_diagnostics import (
    run_reward_diagnostics,
    write_reward_diagnostics,
)


def test_reward_diagnostics_runs_heuristic_baseline() -> None:
    # heuristic学習者のbaseline rewardを複数seedで集計できることを確認する。
    diagnostics = run_reward_diagnostics(seeds=(1, 2), policy="heuristic")

    assert diagnostics.policy == "heuristic"
    assert diagnostics.episode_count == 2
    assert diagnostics.average_step_count > 0
    assert len(diagnostics.episodes) == 2


def test_reward_diagnostics_runs_random_baseline() -> None:
    # random学習者のbaseline rewardも同じ形式で集計できることを確認する。
    diagnostics = run_reward_diagnostics(seeds=(1,), policy="random")

    assert diagnostics.policy == "random"
    assert diagnostics.episode_count == 1
    assert diagnostics.episodes[0].step_count > 0


def test_reward_diagnostics_rejects_empty_seeds() -> None:
    # seedが空の場合は平均値を作らず明示的に失敗することを確認する。
    try:
        run_reward_diagnostics(seeds=())
    except ValueError as error:
        assert "seeds" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_reward_diagnostics_writes_json_and_csv(tmp_path) -> None:
    # reward診断結果を後から確認できるJSON/CSVとして保存できることを確認する。
    diagnostics = run_reward_diagnostics(seeds=(1,), policy="heuristic")
    json_output = tmp_path / "reward.json"
    csv_output = tmp_path / "reward.csv"

    write_reward_diagnostics(
        diagnostics,
        json_output=json_output,
        csv_output=csv_output,
    )

    assert "average_total_reward" in json_output.read_text(encoding="utf-8")
    assert "seed,total_reward" in csv_output.read_text(encoding="utf-8")
