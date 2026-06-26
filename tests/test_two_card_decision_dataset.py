from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    collect_two_card_decision_samples,
    read_two_card_decision_samples,
    summarize_two_card_decision_samples,
    two_card_decision_dataset_summary_to_dict,
    write_two_card_decision_samples,
)


def test_collect_two_card_decision_samples_labels_turn_starts() -> None:
    # 二値判断用の手番開始観測と1枚/2枚ラベルを収集できる。
    samples = collect_two_card_decision_samples(
        source_games=1,
        source_seed_start=1,
        source_state_limit=3,
        horizon_learner_turns=0,
    )

    assert samples
    assert all(len(sample.observation) == OBSERVATION_SIZE for sample in samples)
    assert all(sample.selected_decision in (0, 1) for sample in samples)
    assert all(sample.player_index == 0 for sample in samples)
    assert all(sample.seed == 1 for sample in samples)


def test_collect_two_card_decision_samples_limits_states_per_game() -> None:
    # 1ゲーム由来の状態数を制限して複数seedから収集できる。
    samples = collect_two_card_decision_samples(
        source_games=3,
        source_seed_start=10,
        source_state_limit=3,
        horizon_learner_turns=0,
        max_source_states_per_game=1,
    )

    assert {sample.seed for sample in samples} == {10, 11, 12}


def test_two_card_decision_samples_round_trip_jsonl(tmp_path) -> None:
    # 二値判断サンプルをJSONLへ保存し、同じ内容として読み戻せる。
    sample = TwoCardDecisionSample(
        observation=tuple(0.0 for _ in range(OBSERVATION_SIZE)),
        selected_decision=1,
        stop_self_loss=1.0,
        stop_relative_loss=0.5,
        two_card_self_loss=0.0,
        two_card_relative_loss=-0.5,
        decision_case="heuristic_stops_with_second_available",
        player_index=0,
        hand_count=5,
        seed=1,
        turn_index=0,
    )
    path = tmp_path / "two-card-decision-samples.jsonl"

    write_two_card_decision_samples((sample,), path)
    loaded = read_two_card_decision_samples(path)

    assert loaded == (sample,)


def test_two_card_decision_samples_accept_utf8_bom(tmp_path) -> None:
    # PowerShell等が付けたUTF-8 BOMを含むJSONLも読み込める。
    path = tmp_path / "samples-with-bom.jsonl"
    path.write_text(
        '{"observation":['
        + ",".join("0" for _ in range(OBSERVATION_SIZE))
        + '],"selected_decision":0,"stop_self_loss":0,'
        '"stop_relative_loss":0,"two_card_self_loss":1,'
        '"two_card_relative_loss":1,"decision_case":"test",'
        '"player_index":0,"hand_count":6,"seed":1,"turn_index":0}\n',
        encoding="utf-8-sig",
    )

    loaded = read_two_card_decision_samples(path)

    assert len(loaded) == 1


def test_summarize_two_card_decision_samples_counts_coverage() -> None:
    # 二値判断サンプルの手札枚数、選択ラベル、ケース分類を集計できる。
    samples = (
        TwoCardDecisionSample(
            tuple(0.0 for _ in range(OBSERVATION_SIZE)),
            0,
            0.0,
            0.0,
            1.0,
            1.0,
            "heuristic_stops_with_second_available",
            0,
            6,
            1,
            0,
        ),
        TwoCardDecisionSample(
            tuple(0.1 for _ in range(OBSERVATION_SIZE)),
            1,
            1.0,
            1.0,
            0.0,
            0.0,
            "heuristic_plays_damaging_second",
            0,
            4,
            1,
            1,
        ),
    )

    summary = summarize_two_card_decision_samples(
        samples,
        source_games=1,
        source_states=2,
        horizon_learner_turns=4,
        rollout_count=4,
    )
    summary_dict = two_card_decision_dataset_summary_to_dict(summary)

    assert summary.samples == 2
    assert summary.hand_count_histogram[6] == 1
    assert summary.hand_count_histogram[4] == 1
    assert summary.selected_decision_histogram == (1, 1)
    assert summary.decision_case_histogram["heuristic_plays_damaging_second"] == 1
    assert summary.source_seed_count == 1
    assert summary.rollout_count == 4
    assert summary_dict["selected_decision_histogram"] == [1, 1]
    assert summary_dict["rollout_count"] == 4
