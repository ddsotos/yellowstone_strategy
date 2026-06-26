"""Train a supervised binary one-card/two-card decision model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_decision_dataset import read_two_card_decision_samples


@dataclass(frozen=True, slots=True)
class TwoCardDecisionTrainingResult:
    """Summary of one binary classifier training run."""

    sample_count: int
    train_count: int
    validation_count: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    validation_positive_rate: float
    target: str
    minimum_target_margin: float
    validation_dataset_path: str | None
    output_path: str


def make_two_card_decision_model(torch: Any) -> Any:
    """Create the compact binary decision model architecture."""
    return torch.nn.Sequential(
        torch.nn.Linear(OBSERVATION_SIZE, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


def train_two_card_decision_model(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    validation_ratio: float = 0.2,
    seed: int = 1,
    balance_classes: bool = True,
    target: str = "relative_loss",
    minimum_target_margin: float = 0.0,
    validation_dataset_path: str | Path | None = None,
    train_sample_limit: int | None = None,
) -> TwoCardDecisionTrainingResult:
    """Train an MLP to imitate the four-turn binary rollout decision."""
    torch = _import_torch()
    if minimum_target_margin < 0:
        raise ValueError("minimum_target_margin must not be negative")
    if target not in ("self_loss", "relative_loss"):
        raise ValueError("target must be 'self_loss' or 'relative_loss'")
    all_samples = read_two_card_decision_samples(dataset_path)
    if train_sample_limit is not None:
        if train_sample_limit <= 0:
            raise ValueError("train_sample_limit must be positive")
        all_samples = all_samples[:train_sample_limit]
    train_samples = tuple(
        sample
        for sample in all_samples
        if abs(_target_advantage(sample, target)) >= minimum_target_margin
    )
    if len(train_samples) < 2:
        raise ValueError("at least two samples are required")

    torch.manual_seed(seed)
    if validation_dataset_path is None:
        features, targets = _samples_to_tensors(torch, train_samples, target)
        permutation = torch.randperm(len(train_samples))
        features = features[permutation]
        targets = targets[permutation]
        validation_count = max(1, int(len(train_samples) * validation_ratio))
        train_count = len(train_samples) - validation_count
        if train_count <= 0:
            raise ValueError("validation split leaves no training samples")
        train_x, validation_x = features[:train_count], features[train_count:]
        train_y, validation_y = targets[:train_count], targets[train_count:]
    else:
        validation_samples = tuple(
            sample
            for sample in read_two_card_decision_samples(validation_dataset_path)
            if abs(_target_advantage(sample, target)) >= minimum_target_margin
        )
        if not validation_samples:
            raise ValueError("validation dataset has no eligible samples")
        train_x, train_y = _samples_to_tensors(torch, train_samples, target)
        validation_x, validation_y = _samples_to_tensors(
            torch, validation_samples, target
        )
        train_count = len(train_samples)
        validation_count = len(validation_samples)

    model = make_two_card_decision_model(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weight = _positive_class_weight(
        torch, train_y, enabled=balance_classes
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    for _ in range(epochs):
        model.train()
        batch_permutation = torch.randperm(train_count)
        for start in range(0, train_count, batch_size):
            indices = batch_permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(train_x), train_y).item())
        validation_logits = model(validation_x)
        validation_loss = float(loss_fn(validation_logits, validation_y).item())
        validation_predictions = (torch.sigmoid(validation_logits) >= 0.5).to(
            torch.int64
        )
        confusion = _confusion_matrix(
            validation_y.to(torch.int64), validation_predictions
        )
        accuracy, balanced_accuracy = _classification_accuracy(confusion)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_size": OBSERVATION_SIZE,
            "target": f"four_turn_{target}_decision",
            "decision_target": target,
            "minimum_target_margin": minimum_target_margin,
            "balance_classes": balance_classes,
        },
        output,
    )
    result = TwoCardDecisionTrainingResult(
        sample_count=train_count + validation_count,
        train_count=train_count,
        validation_count=validation_count,
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_accuracy=accuracy,
        validation_balanced_accuracy=balanced_accuracy,
        validation_confusion_matrix=confusion,
        validation_positive_rate=float(validation_y.mean().item()),
        target=target,
        minimum_target_margin=minimum_target_margin,
        validation_dataset_path=(
            None
            if validation_dataset_path is None
            else str(validation_dataset_path)
        ),
        output_path=str(output),
    )
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _positive_class_weight(torch: Any, targets: Any, *, enabled: bool) -> Any:
    if not enabled:
        return torch.tensor([1.0], dtype=torch.float32)
    positive_count = float(targets.sum().item())
    negative_count = float(targets.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)


def _samples_to_tensors(
    torch: Any,
    samples: tuple[Any, ...],
    target: str,
) -> tuple[Any, Any]:
    features = torch.tensor(
        [sample.observation for sample in samples], dtype=torch.float32
    )
    targets = torch.tensor(
        [[int(_target_advantage(sample, target) > 0)] for sample in samples],
        dtype=torch.float32,
    )
    return features, targets


def _target_advantage(sample: Any, target: str) -> float:
    if target == "self_loss":
        return float(sample.stop_self_loss - sample.two_card_self_loss)
    if target == "relative_loss":
        return float(sample.stop_relative_loss - sample.two_card_relative_loss)
    raise ValueError("target must be 'self_loss' or 'relative_loss'")


def _confusion_matrix(
    targets: Any, predictions: Any
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (
            int(((targets == 0) & (predictions == 0)).sum().item()),
            int(((targets == 0) & (predictions == 1)).sum().item()),
        ),
        (
            int(((targets == 1) & (predictions == 0)).sum().item()),
            int(((targets == 1) & (predictions == 1)).sum().item()),
        ),
    )


def _classification_accuracy(
    confusion: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[float, float]:
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    total = true_negative + false_positive + false_negative + true_positive
    accuracy = (true_negative + true_positive) / total if total else 0.0
    recalls = []
    if true_negative + false_positive:
        recalls.append(true_negative / (true_negative + false_positive))
    if true_positive + false_negative:
        recalls.append(true_positive / (true_positive + false_negative))
    return accuracy, sum(recalls) / len(recalls) if recalls else 0.0


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for binary decision training. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch


def main() -> None:
    """CLI entry point for binary decision training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-balance-classes", action="store_true")
    parser.add_argument(
        "--target", choices=("self_loss", "relative_loss"), default="relative_loss"
    )
    parser.add_argument("--minimum-target-margin", type=float, default=0.0)
    parser.add_argument("--validation-dataset-path")
    parser.add_argument("--train-sample-limit", type=int)
    args = parser.parse_args()
    result = train_two_card_decision_model(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        report_path=args.report_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        balance_classes=not args.no_balance_classes,
        target=args.target,
        minimum_target_margin=args.minimum_target_margin,
        validation_dataset_path=args.validation_dataset_path,
        train_sample_limit=args.train_sample_limit,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
