"""Train a small supervised state-value model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.value_dataset import read_state_value_samples


@dataclass(frozen=True, slots=True)
class StateValueTrainingResult:
    """Summary of a supervised state-value training run."""

    sample_count: int
    train_count: int
    validation_count: int
    train_loss: float
    validation_loss: float
    validation_loss_by_hand_count: tuple[float | None, ...]
    validation_count_by_hand_count: tuple[int, ...]
    output_path: str


def make_state_value_model(torch: Any) -> Any:
    """Create the supervised state-value model architecture."""
    return torch.nn.Sequential(
        torch.nn.Linear(OBSERVATION_SIZE, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )


def train_state_value_model(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_ratio: float = 0.2,
    seed: int = 1,
) -> StateValueTrainingResult:
    """Train a compact MLP that predicts final loss share from observations."""
    torch = _import_torch()
    samples = read_state_value_samples(dataset_path)
    if len(samples) < 2:
        raise ValueError("at least two samples are required")

    torch.manual_seed(seed)
    features = torch.tensor(
        [sample.observation for sample in samples],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[sample.target_loss_share] for sample in samples],
        dtype=torch.float32,
    )
    hand_counts = torch.tensor(
        [sample.hand_count for sample in samples],
        dtype=torch.int64,
    )
    permutation = torch.randperm(len(samples))
    features = features[permutation]
    targets = targets[permutation]
    hand_counts = hand_counts[permutation]

    validation_count = max(1, int(len(samples) * validation_ratio))
    train_count = len(samples) - validation_count
    if train_count <= 0:
        raise ValueError("validation split leaves no training samples")

    train_x = features[:train_count]
    train_y = targets[:train_count]
    validation_x = features[train_count:]
    validation_y = targets[train_count:]
    validation_hand_counts = hand_counts[train_count:]

    model = make_state_value_model(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for start in range(0, train_count, batch_size):
            batch_x = train_x[start : start + batch_size]
            batch_y = train_y[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_loss = float(loss_fn(model(train_x), train_y).item())
        validation_predictions = model(validation_x)
        validation_loss = float(
            loss_fn(validation_predictions, validation_y).item()
        )
        validation_loss_by_hand_count, validation_count_by_hand_count = (
            _validation_metrics_by_hand_count(
                torch,
                validation_predictions,
                validation_y,
                validation_hand_counts,
            )
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_size": OBSERVATION_SIZE,
            "target": "final_loss_share",
            "hand_count_target_means": _hand_count_target_means(samples),
        },
        output,
    )

    result = StateValueTrainingResult(
        sample_count=len(samples),
        train_count=train_count,
        validation_count=validation_count,
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_loss_by_hand_count=validation_loss_by_hand_count,
        validation_count_by_hand_count=validation_count_by_hand_count,
        output_path=str(output),
    )
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(state_value_training_result_to_dict(result), indent=2),
            encoding="utf-8",
        )
    return result


def state_value_training_result_to_dict(
    result: StateValueTrainingResult,
) -> dict[str, int | float | str | list[float | None] | list[int]]:
    """Convert training result to JSON-friendly data."""
    return {
        "sample_count": result.sample_count,
        "train_count": result.train_count,
        "validation_count": result.validation_count,
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "validation_loss_by_hand_count": list(result.validation_loss_by_hand_count),
        "validation_count_by_hand_count": list(result.validation_count_by_hand_count),
        "output_path": result.output_path,
    }


def _hand_count_target_means(samples: tuple[Any, ...]) -> list[float]:
    grouped: list[list[float]] = [[] for _ in range(7)]
    all_targets: list[float] = []
    for sample in samples:
        target = float(sample.target_loss_share)
        all_targets.append(target)
        if 0 <= sample.hand_count < len(grouped):
            grouped[sample.hand_count].append(target)
    fallback = sum(all_targets) / len(all_targets)
    return [
        (sum(targets) / len(targets)) if targets else fallback
        for targets in grouped
    ]


def _validation_metrics_by_hand_count(
    torch: Any,
    predictions: Any,
    targets: Any,
    hand_counts: Any,
) -> tuple[tuple[float | None, ...], tuple[int, ...]]:
    losses: list[float | None] = []
    counts: list[int] = []
    squared_errors = torch.square(predictions - targets).flatten()
    for hand_count in range(7):
        mask = hand_counts == hand_count
        count = int(mask.sum().item())
        counts.append(count)
        if count == 0:
            losses.append(None)
            continue
        losses.append(float(squared_errors[mask].mean().item()))
    return tuple(losses), tuple(counts)


class _StateValueModel:
    def __new__(cls, torch: Any) -> Any:
        return make_state_value_model(torch)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for value model training. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch


def main() -> None:
    """CLI entry point for supervised state-value training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    result = train_state_value_model(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        report_path=args.report_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(state_value_training_result_to_dict(result), indent=2))


if __name__ == "__main__":
    main()
