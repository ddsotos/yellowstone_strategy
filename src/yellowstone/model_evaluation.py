"""Evaluate a trained MaskablePPO model against baseline bots."""

from __future__ import annotations

import argparse
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


def model_evaluation_dependencies_available() -> bool:
    """Return whether optional model evaluation dependencies are importable."""
    return gymnasium_available() and np is not None and MaskablePPO is not None


def _require_model_evaluation_dependencies() -> None:
    if not model_evaluation_dependencies_available():
        raise ImportError(
            "RL model evaluation dependencies are not installed. "
            'Install them with: python -m pip install -e ".[rl]"'
        )


def _parse_args() -> tuple[Path, tuple[int, ...], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    return args.model_path, seeds, not args.stochastic


def _format_summary(name: str, summary: EvaluationSummary) -> str:
    return (
        f"{name}: matches={summary.match_count} "
        f"win_rates={summary.win_rates} "
        f"avg_loss={summary.average_loss_scores} "
        f"avg_turns={summary.average_turn_count:.2f}"
    )


def main() -> None:
    """Evaluate a saved model from command-line arguments."""
    model_path, seeds, deterministic = _parse_args()
    result = evaluate_model(
        model_path,
        seeds=seeds,
        deterministic=deterministic,
    )
    print(_format_summary("model_vs_heuristic", result.model_vs_heuristic))
    print(_format_summary("heuristic_only", result.heuristic_only))
    print(_format_summary("random_only", result.random_only))


if __name__ == "__main__":
    main()
