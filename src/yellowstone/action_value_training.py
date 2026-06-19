"""Train a supervised action-value model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from yellowstone.action_value_dataset import read_action_value_samples
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.turn_action_space import TURN_ACTION_SPACE_SIZE


ActionValueTarget = Literal[
    "relative_loss",
    "self_loss",
    "relative_advantage",
    "self_advantage",
]
ACTION_VALUE_FEATURE_SIZE = OBSERVATION_SIZE * 2 + TURN_ACTION_SPACE_SIZE


@dataclass(frozen=True, slots=True)
class ActionValueTrainingResult:
    """Summary of an action-value training run."""

    sample_count: int
    train_count: int
    validation_count: int
    train_loss: float
    validation_loss: float
    target: ActionValueTarget
    output_path: str


def make_action_value_model(torch: Any) -> Any:
    """Create a compact MLP for Q(s,a)-style value prediction."""
    return torch.nn.Sequential(
        torch.nn.Linear(ACTION_VALUE_FEATURE_SIZE, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 48),
        torch.nn.ReLU(),
        torch.nn.Linear(48, 1),
    )


def train_action_value_model(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    target: ActionValueTarget = "relative_loss",
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_ratio: float = 0.2,
    seed: int = 1,
) -> ActionValueTrainingResult:
    """Train a supervised action-value model from JSONL samples."""
    torch = _import_torch()
    samples = read_action_value_samples(dataset_path)
    if len(samples) < 2:
        raise ValueError("at least two samples are required")

    torch.manual_seed(seed)
    features = torch.tensor(
        [
            _feature_vector(
                sample.observation,
                sample.after_observation,
                sample.action_index,
            )
            for sample in samples
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[_target_value(sample, target)] for sample in samples],
        dtype=torch.float32,
    )
    permutation = torch.randperm(len(samples))
    features = features[permutation]
    targets = targets[permutation]

    validation_count = max(1, int(len(samples) * validation_ratio))
    train_count = len(samples) - validation_count
    if train_count <= 0:
        raise ValueError("validation split leaves no training samples")

    train_x = features[:train_count]
    train_y = targets[:train_count]
    validation_x = features[train_count:]
    validation_y = targets[train_count:]

    model = make_action_value_model(torch)
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
        validation_loss = float(loss_fn(model(validation_x), validation_y).item())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_size": ACTION_VALUE_FEATURE_SIZE,
            "observation_size": OBSERVATION_SIZE,
            "action_space_size": TURN_ACTION_SPACE_SIZE,
            "target": target,
        },
        output,
    )

    result = ActionValueTrainingResult(
        sample_count=len(samples),
        train_count=train_count,
        validation_count=validation_count,
        train_loss=train_loss,
        validation_loss=validation_loss,
        target=target,
        output_path=str(output),
    )
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(action_value_training_result_to_dict(result), indent=2),
            encoding="utf-8",
        )
    return result


def action_value_training_result_to_dict(
    result: ActionValueTrainingResult,
) -> dict[str, int | float | str]:
    """Convert training result to JSON-friendly data."""
    return {
        "sample_count": result.sample_count,
        "train_count": result.train_count,
        "validation_count": result.validation_count,
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "target": result.target,
        "output_path": result.output_path,
    }


def _feature_vector(
    observation: tuple[float, ...],
    after_observation: tuple[float, ...],
    action_index: int,
) -> tuple[float, ...]:
    action_features = [0.0] * TURN_ACTION_SPACE_SIZE
    action_features[action_index] = 1.0
    return observation + after_observation + tuple(action_features)


def _target_value(sample: Any, target: ActionValueTarget) -> float:
    if target == "relative_loss":
        return float(sample.target_relative_loss)
    if target == "self_loss":
        return float(sample.target_self_loss)
    if target == "relative_advantage":
        return float(sample.target_relative_advantage)
    if target == "self_advantage":
        return float(sample.target_self_advantage)
    raise ValueError(f"unsupported target: {target}")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for action-value training. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch


def main() -> None:
    """CLI entry point for action-value model training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument(
        "--target",
        choices=(
            "relative_loss",
            "self_loss",
            "relative_advantage",
            "self_advantage",
        ),
        default="relative_loss",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    result = train_action_value_model(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        report_path=args.report_path,
        target=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(action_value_training_result_to_dict(result), indent=2))


if __name__ == "__main__":
    main()
