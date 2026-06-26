import yellowstone.two_card_advantage_diagnostics as diagnostics_module
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_advantage_diagnostics import diagnose_advantage_thresholds
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    write_two_card_decision_samples,
)


def test_diagnostics_measure_only_heuristic_card_count_changes(
    tmp_path, monkeypatch
) -> None:
    # 閾値を超えてheuristicと異なる枚数を選んだ局面だけを診断する。
    samples = (
        _sample(decision_case="heuristic_stops_with_second_available", advantage=2.0),
        _sample(decision_case="heuristic_plays_no_damage_second", advantage=-1.0),
        _sample(decision_case="heuristic_stops_with_second_available", advantage=3.0),
    )
    dataset_path = tmp_path / "samples.jsonl"
    write_two_card_decision_samples(samples, dataset_path)
    predictions = iter((2.5, -1.5, 0.5))
    monkeypatch.setattr(
        diagnostics_module,
        "predict_two_card_advantage_from_observation",
        lambda observation, model_path: next(predictions),
    )

    diagnostic = diagnose_advantage_thresholds(
        tmp_path / "model.pt",
        dataset_path,
        thresholds=(1.0,),
    )[0]

    assert diagnostic.changed_count == 2
    assert diagnostic.changed_rate == 2 / 3
    assert diagnostic.changed_accuracy == 1.0
    assert diagnostic.mean_target_improvement == 1.0


def _sample(*, decision_case: str, advantage: float) -> TwoCardDecisionSample:
    return TwoCardDecisionSample(
        observation=tuple(0.0 for _ in range(OBSERVATION_SIZE)),
        selected_decision=int(advantage > 0),
        stop_self_loss=advantage,
        stop_relative_loss=advantage,
        two_card_self_loss=0.0,
        two_card_relative_loss=0.0,
        decision_case=decision_case,
        player_index=0,
        hand_count=6,
        seed=1,
        turn_index=0,
    )
