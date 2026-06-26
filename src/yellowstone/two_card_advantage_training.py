"""Train a regression model for one-card/two-card rollout advantage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    read_two_card_decision_samples,
)


@dataclass(frozen=True, slots=True)
class TwoCardAdvantageTrainingResult:
    """Summary of one advantage-regression training run."""

    train_count: int
    validation_count: int
    train_loss: float
    validation_loss: float
    validation_mae: float
    validation_rmse: float
    validation_sign_accuracy: float
    validation_balanced_sign_accuracy: float
    validation_heuristic_sign_accuracy: float
    target: str
    target_mean: float
    target_std: float
    validation_dataset_path: str | None
    output_path: str
    best_epoch: int
    epochs_trained: int
    decision_case_filter: tuple[str, ...] | None


def make_two_card_advantage_model(torch: Any) -> Any:
    """Create the compact advantage-regression architecture."""
    return torch.nn.Sequential(
        torch.nn.Linear(OBSERVATION_SIZE, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


def train_two_card_advantage_model(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    validation_dataset_path: str | Path | None = None,
    train_sample_limit: int | None = None,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    validation_ratio: float = 0.2,
    seed: int = 1,
    target: str = "relative_loss",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    additional_dataset_paths: tuple[str | Path, ...] = (),
    decision_case_filter: tuple[str, ...] | None = None,
) -> TwoCardAdvantageTrainingResult:
    """Train a Huber-loss model that predicts rollout loss difference."""
    torch = _import_torch()
    if target not in ("self_loss", "relative_loss"):
        raise ValueError("target must be 'self_loss' or 'relative_loss'")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must not be negative")
    train_samples = read_two_card_decision_samples(dataset_path) + tuple(
        sample
        for additional_path in additional_dataset_paths
        for sample in read_two_card_decision_samples(additional_path)
    )
    train_samples = _filter_samples_by_decision_case(
        train_samples, decision_case_filter
    )
    if train_sample_limit is not None:
        if train_sample_limit <= 0:
            raise ValueError("train_sample_limit must be positive")
        train_samples = train_samples[:train_sample_limit]
    if len(train_samples) < 2:
        raise ValueError("at least two training samples are required")

    torch.manual_seed(seed)
    if validation_dataset_path is None:
        validation_count = max(1, int(len(train_samples) * validation_ratio))
        permutation = torch.randperm(len(train_samples)).tolist()
        shuffled = tuple(train_samples[index] for index in permutation)
        validation_samples = shuffled[-validation_count:]
        train_samples = shuffled[:-validation_count]
    else:
        validation_samples = _filter_samples_by_decision_case(
            read_two_card_decision_samples(validation_dataset_path),
            decision_case_filter,
        )
    if not validation_samples:
        raise ValueError("validation dataset must not be empty")

    train_targets_raw = tuple(_target_advantage(sample, target) for sample in train_samples)
    target_mean = sum(train_targets_raw) / len(train_targets_raw)
    target_variance = sum(
        (value - target_mean) ** 2 for value in train_targets_raw
    ) / len(train_targets_raw)
    target_std = max(sqrt(target_variance), 1e-6)
    train_x, train_y = _samples_to_tensors(
        torch, train_samples, target, target_mean, target_std
    )
    validation_x, validation_y = _samples_to_tensors(
        torch, validation_samples, target, target_mean, target_std
    )

    model = make_two_card_advantage_model(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.HuberLoss()
    train_count = len(train_samples)
    best_epoch = epochs
    epochs_trained = 0
    best_validation_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    for epoch_index in range(epochs):
        model.train()
        permutation = torch.randperm(train_count)
        for start in range(0, train_count, batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
        epochs_trained = epoch_index + 1
        if early_stopping_patience is None:
            continue
        model.eval()
        with torch.no_grad():
            epoch_validation_loss = float(
                loss_fn(model(validation_x), validation_y).item()
            )
        if epoch_validation_loss < best_validation_loss - early_stopping_min_delta:
            best_validation_loss = epoch_validation_loss
            best_epoch = epochs_trained
            best_state_dict = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        best_epoch = epochs_trained

    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(train_x), train_y).item())
        validation_predictions_scaled = model(validation_x)
        validation_loss = float(
            loss_fn(validation_predictions_scaled, validation_y).item()
        )
        validation_predictions = (
            validation_predictions_scaled * target_std + target_mean
        ).flatten()
        validation_targets = (
            validation_y * target_std + target_mean
        ).flatten()
        errors = validation_predictions - validation_targets
        validation_mae = float(torch.abs(errors).mean().item())
        validation_rmse = float(torch.sqrt(torch.square(errors).mean()).item())
        sign_accuracy, balanced_sign_accuracy = _sign_accuracy(
            validation_predictions, validation_targets
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_size": OBSERVATION_SIZE,
            "target": target,
            "target_mean": target_mean,
            "target_std": target_std,
            "best_epoch": best_epoch,
        },
        output,
    )
    result = TwoCardAdvantageTrainingResult(
        train_count=train_count,
        validation_count=len(validation_samples),
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_mae=validation_mae,
        validation_rmse=validation_rmse,
        validation_sign_accuracy=sign_accuracy,
        validation_balanced_sign_accuracy=balanced_sign_accuracy,
        validation_heuristic_sign_accuracy=_heuristic_sign_accuracy(
            validation_samples, target
        ),
        target=target,
        target_mean=target_mean,
        target_std=target_std,
        validation_dataset_path=(
            None
            if validation_dataset_path is None
            else str(validation_dataset_path)
        ),
        output_path=str(output),
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        decision_case_filter=decision_case_filter,
    )
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _samples_to_tensors(
    torch: Any,
    samples: tuple[TwoCardDecisionSample, ...],
    target: str,
    target_mean: float,
    target_std: float,
) -> tuple[Any, Any]:
    features = torch.tensor(
        [sample.observation for sample in samples], dtype=torch.float32
    )
    targets = torch.tensor(
        [
            [(_target_advantage(sample, target) - target_mean) / target_std]
            for sample in samples
        ],
        dtype=torch.float32,
    )
    return features, targets


def _filter_samples_by_decision_case(
    samples: tuple[TwoCardDecisionSample, ...],
    decision_case_filter: tuple[str, ...] | None,
) -> tuple[TwoCardDecisionSample, ...]:
    if decision_case_filter is None:
        return samples
    allowed = set(decision_case_filter)
    return tuple(sample for sample in samples if sample.decision_case in allowed)


def _target_advantage(sample: TwoCardDecisionSample, target: str) -> float:
    if target == "self_loss":
        return sample.stop_self_loss - sample.two_card_self_loss
    return sample.stop_relative_loss - sample.two_card_relative_loss


def _sign_accuracy(predictions: Any, targets: Any) -> tuple[float, float]:
    predicted_positive = predictions > 0
    target_positive = targets > 0
    accuracy = float((predicted_positive == target_positive).float().mean().item())
    recalls: list[float] = []
    for expected in (False, True):
        mask = target_positive == expected
        if int(mask.sum().item()) > 0:
            recalls.append(
                float((predicted_positive[mask] == expected).float().mean().item())
            )
    return accuracy, sum(recalls) / len(recalls)


def _heuristic_sign_accuracy(
    samples: tuple[TwoCardDecisionSample, ...], target: str
) -> float:
    correct = 0
    for sample in samples:
        heuristic_two = sample.decision_case != "heuristic_stops_with_second_available"
        target_two = _target_advantage(sample, target) > 0
        correct += int(heuristic_two == target_two)
    return correct / len(samples)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for advantage regression. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch


def main() -> None:
    """CLI entry point for advantage regression."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--validation-dataset-path")
    parser.add_argument("--train-sample-limit", type=int)
    parser.add_argument("--additional-dataset-path", action="append", default=[])
    parser.add_argument("--decision-case-filter", action="append")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--target", choices=("self_loss", "relative_loss"), default="relative_loss"
    )
    args = parser.parse_args()
    result = train_two_card_advantage_model(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        report_path=args.report_path,
        validation_dataset_path=args.validation_dataset_path,
        train_sample_limit=args.train_sample_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        target=args.target,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        additional_dataset_paths=tuple(args.additional_dataset_path),
        decision_case_filter=(
            None
            if args.decision_case_filter is None
            else tuple(args.decision_case_filter)
        ),
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
