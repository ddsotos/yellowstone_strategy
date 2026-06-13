"""Evaluate a trained MaskablePPO model against baseline bots."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from yellowstone.action_space import action_from_index, legal_action_mask
from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.evaluation import (
    EvaluationSummary,
    evaluate_policies,
    make_heuristic_policies,
    make_random_policies,
)
from yellowstone.gym_env import gymnasium_available
from yellowstone.observation import state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.types import Action, GameState

try:
    import numpy as np
    from sb3_contrib import MaskablePPO
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    MaskablePPO = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """Evaluation summaries for a learned model and baselines."""

    model_vs_heuristic: EvaluationSummary
    heuristic_only: EvaluationSummary
    random_only: EvaluationSummary


@dataclass(frozen=True, slots=True)
class LearnedModelBot:
    """BotPolicy adapter for a MaskablePPO model."""

    model: object
    deterministic: bool = True

    def choose_action(self, state: GameState) -> Action | None:
        """Choose a legal action using the trained model and action mask."""
        mask = legal_action_mask(state)
        if not any(mask):
            return None
        observation = np.asarray(
            normalize_observation(state_to_observation(state)),
            dtype=np.float32,
        )
        action_index, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=np.asarray(mask, dtype=np.bool_),
        )
        return action_from_index(int(action_index), state)


def evaluate_model(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    deterministic: bool = True,
) -> ModelEvaluationResult:
    """Evaluate a saved model against heuristic NPCs and baselines."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    _require_model_evaluation_dependencies()
    model = MaskablePPO.load(model_path)
    learned_policy: BotPolicy = LearnedModelBot(
        model=model,
        deterministic=deterministic,
    )
    model_policies: tuple[BotPolicy, ...] = (
        learned_policy,
        HeuristicBot(),
        HeuristicBot(),
        HeuristicBot(),
    )
    model_summary, _ = evaluate_policies(model_policies, seeds=seeds)
    heuristic_summary, _ = evaluate_policies(make_heuristic_policies(), seeds=seeds)
    random_summary, _ = evaluate_policies(
        make_random_policies(seed=seeds[0]),
        seeds=seeds,
    )
    return ModelEvaluationResult(
        model_vs_heuristic=model_summary,
        heuristic_only=heuristic_summary,
        random_only=random_summary,
    )


def evaluation_summary_to_dict(summary: EvaluationSummary) -> dict[str, object]:
    """Convert an EvaluationSummary into JSON-friendly data."""
    return asdict(summary)


def model_evaluation_result_to_dict(
    result: ModelEvaluationResult,
) -> dict[str, dict[str, object]]:
    """Convert model evaluation summaries into JSON-friendly data."""
    return {
        "model_vs_heuristic": evaluation_summary_to_dict(result.model_vs_heuristic),
        "heuristic_only": evaluation_summary_to_dict(result.heuristic_only),
        "random_only": evaluation_summary_to_dict(result.random_only),
    }


def write_model_evaluation_result(
    result: ModelEvaluationResult,
    *,
    json_output: Path | None = None,
    csv_output: Path | None = None,
) -> None:
    """Write evaluation results as JSON and/or CSV."""
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(
                model_evaluation_result_to_dict(result),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "scenario",
                    "match_count",
                    "win_rates",
                    "average_loss_scores",
                    "average_turn_count",
                    "total_elapsed_seconds",
                    "average_elapsed_seconds",
                ],
            )
            writer.writeheader()
            for scenario, summary in model_evaluation_result_to_dict(result).items():
                writer.writerow({"scenario": scenario, **summary})


def model_evaluation_dependencies_available() -> bool:
    """Return whether optional model evaluation dependencies are importable."""
    return gymnasium_available() and np is not None and MaskablePPO is not None


def _require_model_evaluation_dependencies() -> None:
    if not model_evaluation_dependencies_available():
        raise ImportError(
            "RL model evaluation dependencies are not installed. "
            'Install them with: python -m pip install -e ".[rl]"'
        )


def _parse_args() -> tuple[Path, tuple[int, ...], bool, Path | None, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    return args.model_path, seeds, not args.stochastic, args.json_output, args.csv_output


def _format_summary(name: str, summary: EvaluationSummary) -> str:
    return (
        f"{name}: matches={summary.match_count} "
        f"win_rates={summary.win_rates} "
        f"avg_loss={summary.average_loss_scores} "
        f"avg_turns={summary.average_turn_count:.2f}"
    )


def main() -> None:
    """Evaluate a saved model from command-line arguments."""
    model_path, seeds, deterministic, json_output, csv_output = _parse_args()
    result = evaluate_model(
        model_path,
        seeds=seeds,
        deterministic=deterministic,
    )
    write_model_evaluation_result(
        result,
        json_output=json_output,
        csv_output=csv_output,
    )
    print(_format_summary("model_vs_heuristic", result.model_vs_heuristic))
    print(_format_summary("heuristic_only", result.heuristic_only))
    print(_format_summary("random_only", result.random_only))


if __name__ == "__main__":
    main()
