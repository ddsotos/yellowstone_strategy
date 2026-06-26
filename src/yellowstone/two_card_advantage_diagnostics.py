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


@dataclass(frozen=True, slots=True)
class AdvantageThresholdDiagnostic:
    """Policy-change quality at one absolute advantage threshold."""

    threshold: float
    sample_count: int
    changed_count: int
    changed_rate: float
    changed_accuracy: float
    mean_target_improvement: float


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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
