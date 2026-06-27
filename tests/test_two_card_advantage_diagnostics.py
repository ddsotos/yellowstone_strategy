import yellowstone.two_card_advantage_diagnostics as diagnostics_module
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.observation import state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.two_card_advantage_diagnostics import diagnose_advantage_thresholds
from yellowstone.two_card_advantage_diagnostics import (
    diagnose_one_to_two_override_buckets,
)
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    write_two_card_decision_samples,
)
from yellowstone.types import GameState, PlayerState


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


def test_bucket_diagnostics_groups_one_to_two_overrides(
    tmp_path, monkeypatch
) -> None:
    # 1→2へ変更する局面だけを、即時差分やrankでbucket集計できる。
    samples = (
        _sample(
            decision_case="heuristic_stops_with_second_available",
            advantage=2.0,
            observation=_observation(
                bonuses=(1, 2),
                negative_deltas=(0, 0),
                played_ranks=(2, 1, 3),
            ),
        ),
        _sample(
            decision_case="heuristic_stops_with_second_available",
            advantage=-1.0,
            observation=_observation(
                bonuses=(1, 1),
                negative_deltas=(0, 1),
                played_ranks=(4, 4, 6),
            ),
        ),
        _sample(
            decision_case="heuristic_plays_no_damage_second",
            advantage=3.0,
            observation=_observation(
                bonuses=(0, 4),
                negative_deltas=(0, 0),
                played_ranks=(0, 0, 1),
            ),
        ),
    )
    dataset_path = tmp_path / "samples.jsonl"
    write_two_card_decision_samples(samples, dataset_path)
    predictions = iter((1.0, 0.8, 1.2))
    monkeypatch.setattr(
        diagnostics_module,
        "predict_two_card_advantage_from_observation",
        lambda observation, model_path: next(predictions),
    )

    diagnostics = diagnose_one_to_two_override_buckets(
        tmp_path / "model.pt",
        dataset_path,
        threshold=0.0,
    )
    by_key = {
        (diagnostic.dimension, diagnostic.bucket): diagnostic
        for diagnostic in diagnostics
    }

    assert by_key[("bonus_vs_negative", "bonus_gt_negative")].changed_count == 1
    assert by_key[("bonus_vs_negative", "bonus_gt_negative")].changed_accuracy == 1.0
    assert by_key[("extra_negative", "yes")].changed_accuracy == 0.0
    assert by_key[("two_card_rank_pair", "2-4")].mean_target_improvement_per_change == 2.0


def _sample(
    *,
    decision_case: str,
    advantage: float,
    observation: tuple[float, ...] | None = None,
) -> TwoCardDecisionSample:
    return TwoCardDecisionSample(
        observation=(
            tuple(0.0 for _ in range(OBSERVATION_SIZE))
            if observation is None
            else observation
        ),
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


def _observation(
    *,
    bonuses: tuple[int, int],
    negative_deltas: tuple[int, int],
    played_ranks: tuple[int, int, int],
) -> tuple[float, ...]:
    return normalize_observation(
        state_to_observation(
            GameState(
                players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
            ),
            heuristic_bonuses=bonuses,
            heuristic_negative_deltas=negative_deltas,
            heuristic_played_ranks=played_ranks,
        )
    )
