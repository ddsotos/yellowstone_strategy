"""Validation diagnostics for learned two-card advantage thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from yellowstone.two_card_advantage_model import (
    predict_two_card_advantage_from_observation,
)
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    read_two_card_decision_samples,
)
from yellowstone.observation import (
    BOARD_OBSERVATION_SIZE,
    HAND_OBSERVATION_SIZE,
    HEURISTIC_BONUS_OBSERVATION_SIZE,
    HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE,
    HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE,
    OPPONENT_LAST_TURN_OBSERVATION_SIZE,
    PLAYERS_OBSERVATION_SIZE,
)
from yellowstone.observation_normalization import observation_high_values


@dataclass(frozen=True, slots=True)
class AdvantageThresholdDiagnostic:
    """Policy-change quality at one absolute advantage threshold."""

    threshold: float
    sample_count: int
    changed_count: int
    changed_rate: float
    changed_accuracy: float
    mean_target_improvement: float


@dataclass(frozen=True, slots=True)
class AdvantageBucketDiagnostic:
    """Quality for one interpretable one-to-two override bucket."""

    dimension: str
    bucket: str
    changed_count: int
    changed_accuracy: float
    mean_target_improvement_per_change: float
    mean_prediction: float


def diagnose_advantage_thresholds(
    model_path: str | Path,
    dataset_path: str | Path,
    *,
    thresholds: tuple[float, ...],
    target: str = "relative_loss",
) -> tuple[AdvantageThresholdDiagnostic, ...]:
    """Measure actual heuristic overrides for each prediction threshold."""
    if target not in ("self_loss", "relative_loss"):
        raise ValueError("target must be 'self_loss' or 'relative_loss'")
    if not thresholds or any(threshold < 0 for threshold in thresholds):
        raise ValueError("thresholds must contain non-negative values")
    samples = read_two_card_decision_samples(dataset_path)
    predictions = tuple(
        predict_two_card_advantage_from_observation(sample.observation, model_path)
        for sample in samples
    )
    return tuple(
        _diagnose_threshold(samples, predictions, threshold, target)
        for threshold in thresholds
    )


def diagnose_one_to_two_override_buckets(
    model_path: str | Path,
    dataset_path: str | Path,
    *,
    threshold: float,
    target: str = "relative_loss",
    min_count: int = 1,
) -> tuple[AdvantageBucketDiagnostic, ...]:
    """Summarize predicted one-to-two overrides by simple feature buckets."""
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    if target not in ("self_loss", "relative_loss"):
        raise ValueError("target must be 'self_loss' or 'relative_loss'")
    samples = read_two_card_decision_samples(dataset_path)
    bucket_totals: dict[tuple[str, str], list[float]] = {}
    bucket_correct: dict[tuple[str, str], int] = {}
    bucket_predictions: dict[tuple[str, str], list[float]] = {}
    for sample in samples:
        if sample.decision_case != "heuristic_stops_with_second_available":
            continue
        prediction = predict_two_card_advantage_from_observation(
            sample.observation,
            model_path,
        )
        if prediction <= threshold:
            continue
        advantage = _target_advantage(sample, target)
        features = _override_features(sample)
        for dimension, bucket in _feature_buckets(features):
            key = (dimension, bucket)
            bucket_totals.setdefault(key, []).append(advantage)
            bucket_correct[key] = bucket_correct.get(key, 0) + int(advantage > 0)
            bucket_predictions.setdefault(key, []).append(prediction)
    diagnostics: list[AdvantageBucketDiagnostic] = []
    for key, improvements in bucket_totals.items():
        if len(improvements) < min_count:
            continue
        predictions = bucket_predictions[key]
        diagnostics.append(
            AdvantageBucketDiagnostic(
                dimension=key[0],
                bucket=key[1],
                changed_count=len(improvements),
                changed_accuracy=bucket_correct[key] / len(improvements),
                mean_target_improvement_per_change=(
                    sum(improvements) / len(improvements)
                ),
                mean_prediction=sum(predictions) / len(predictions),
            )
        )
    return tuple(
        sorted(
            diagnostics,
            key=lambda diagnostic: (
                diagnostic.dimension,
                -diagnostic.changed_count,
                diagnostic.bucket,
            ),
        )
    )


def _diagnose_threshold(
    samples: tuple[TwoCardDecisionSample, ...],
    predictions: tuple[float, ...],
    threshold: float,
    target: str,
) -> AdvantageThresholdDiagnostic:
    changed_count = 0
    changed_correct = 0
    total_improvement = 0.0
    for sample, prediction in zip(samples, predictions, strict=True):
        heuristic_two = (
            sample.decision_case != "heuristic_stops_with_second_available"
        )
        model_two = prediction > 0
        if abs(prediction) <= threshold or model_two == heuristic_two:
            continue
        advantage = _target_advantage(sample, target)
        changed_count += 1
        changed_correct += int(model_two == (advantage > 0))
        total_improvement += advantage if model_two else -advantage
    sample_count = len(samples)
    return AdvantageThresholdDiagnostic(
        threshold=threshold,
        sample_count=sample_count,
        changed_count=changed_count,
        changed_rate=0.0 if sample_count == 0 else changed_count / sample_count,
        changed_accuracy=(
            0.0 if changed_count == 0 else changed_correct / changed_count
        ),
        mean_target_improvement=(
            0.0 if sample_count == 0 else total_improvement / sample_count
        ),
    )


@dataclass(frozen=True, slots=True)
class _OverrideFeatures:
    one_card_bonus: int
    two_card_bonus: int
    one_card_negative_delta: int
    two_card_negative_delta: int
    one_card_rank: int
    two_card_low_rank: int
    two_card_high_rank: int
    hand_count: int

    @property
    def bonus_diff(self) -> int:
        return self.two_card_bonus - self.one_card_bonus

    @property
    def negative_diff(self) -> int:
        return self.two_card_negative_delta - self.one_card_negative_delta


def _override_features(sample: TwoCardDecisionSample) -> _OverrideFeatures:
    highs = observation_high_values()
    start = (
        BOARD_OBSERVATION_SIZE
        + HAND_OBSERVATION_SIZE
        + PLAYERS_OBSERVATION_SIZE
        + OPPONENT_LAST_TURN_OBSERVATION_SIZE
    )
    bonus_values = _denormalized_ints(
        sample.observation,
        highs,
        start,
        HEURISTIC_BONUS_OBSERVATION_SIZE,
    )
    negative_start = start + HEURISTIC_BONUS_OBSERVATION_SIZE
    negative_values = _denormalized_ints(
        sample.observation,
        highs,
        negative_start,
        HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE,
    )
    rank_start = negative_start + HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE
    rank_values = _denormalized_ints(
        sample.observation,
        highs,
        rank_start,
        HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE,
    )
    return _OverrideFeatures(
        one_card_bonus=bonus_values[0],
        two_card_bonus=bonus_values[1],
        one_card_negative_delta=negative_values[0],
        two_card_negative_delta=negative_values[1],
        one_card_rank=rank_values[0],
        two_card_low_rank=rank_values[1],
        two_card_high_rank=rank_values[2],
        hand_count=sample.hand_count,
    )


def _denormalized_ints(
    observation: tuple[float, ...],
    highs: tuple[int, ...],
    start: int,
    count: int,
) -> tuple[int, ...]:
    return tuple(
        int(round(observation[index] * highs[index]))
        for index in range(start, start + count)
    )


def _feature_buckets(
    features: _OverrideFeatures,
) -> tuple[tuple[str, str], ...]:
    return (
        ("bonus_diff", _small_int_bucket(features.bonus_diff, high_label="2+")),
        (
            "negative_diff",
            _small_int_bucket(features.negative_diff, high_label="2+"),
        ),
        (
            "bonus_vs_negative",
            (
                "bonus_gt_negative"
                if features.bonus_diff > features.negative_diff
                else "bonus_le_negative"
            ),
        ),
        (
            "extra_negative",
            "yes" if features.negative_diff > 0 else "no",
        ),
        ("hand_count", str(features.hand_count)),
        ("one_card_rank", str(features.one_card_rank + 1)),
        (
            "two_card_rank_pair",
            f"{features.two_card_low_rank + 1}-{features.two_card_high_rank + 1}",
        ),
        (
            "two_card_low_rank",
            str(features.two_card_low_rank + 1),
        ),
        (
            "two_card_high_rank",
            str(features.two_card_high_rank + 1),
        ),
    )


def _small_int_bucket(value: int, *, high_label: str) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    return high_label


def _target_advantage(sample: TwoCardDecisionSample, target: str) -> float:
    if target == "self_loss":
        return sample.stop_self_loss - sample.two_card_self_loss
    return sample.stop_relative_loss - sample.two_card_relative_loss


def main() -> None:
    """CLI entry point for threshold diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("dataset_path")
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument(
        "--target", choices=("self_loss", "relative_loss"), default="relative_loss"
    )
    parser.add_argument("--json-output")
    parser.add_argument("--bucket-json-output")
    parser.add_argument("--bucket-min-count", type=int, default=1)
    args = parser.parse_args()
    diagnostics = diagnose_advantage_thresholds(
        args.model_path,
        args.dataset_path,
        thresholds=tuple(args.threshold),
        target=args.target,
    )
    result = [asdict(diagnostic) for diagnostic in diagnostics]
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.bucket_json_output:
        bucket_output = Path(args.bucket_json_output)
        bucket_output.parent.mkdir(parents=True, exist_ok=True)
        bucket_diagnostics = diagnose_one_to_two_override_buckets(
            args.model_path,
            args.dataset_path,
            threshold=min(args.threshold),
            target=args.target,
            min_count=args.bucket_min_count,
        )
        bucket_output.write_text(
            json.dumps(
                [asdict(diagnostic) for diagnostic in bucket_diagnostics],
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
