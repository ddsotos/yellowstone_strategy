"""Evaluate a supervised action-value model as a greedy policy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from yellowstone.action_value_model import predict_action_values
from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.evaluation import (
    EvaluationSummary,
    evaluate_policies,
    make_heuristic_policies,
    make_random_policies,
)
from yellowstone.model_evaluation import evaluation_summary_to_dict
from yellowstone.game import apply_known_legal_action
from yellowstone.turn_action_space import (
    legal_turn_action_indices,
    resolve_turn_action,
    resolve_turn_action_before_refill,
)
from yellowstone.types import Action, GameState


@dataclass(frozen=True, slots=True)
class ActionValueEvaluationResult:
    """Evaluation summaries for a greedy action-value policy."""

    model_vs_heuristic: EvaluationSummary
    heuristic_only: EvaluationSummary
    random_only: EvaluationSummary


@dataclass(slots=True)
class ActionValueBot:
    """BotPolicy adapter for a supervised action-value model."""

    model_path: Path
    immediate_loss_penalty: float = 0.0
    pending_actions: tuple[Action, ...] = ()

    def choose_action(self, state: GameState) -> Action | None:
        """Choose the legal turn action with the lowest predicted value."""
        if self.pending_actions:
            action = self.pending_actions[0]
            self.pending_actions = self.pending_actions[1:]
            return action

        action_indices = legal_turn_action_indices(state)
        if not action_indices:
            return None
        values = predict_action_values(
            state,
            action_indices=action_indices,
            model_path=self.model_path,
        )
        best_action_index, _ = min(
            (
                (
                    action_index,
                    value
                    + self.immediate_loss_penalty
                    * _immediate_loss_delta(state, action_index),
                )
                for action_index, value in zip(action_indices, values, strict=True)
            ),
            key=lambda item: (item[1], item[0]),
        )
        actions = resolve_turn_action(state, best_action_index)
        if not actions:
            return None
        self.pending_actions = actions[1:]
        return actions[0]


def evaluate_action_value_model(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    immediate_loss_penalty: float = 0.0,
) -> ActionValueEvaluationResult:
    """Evaluate a greedy action-value policy against heuristic NPCs."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    action_value_policy: BotPolicy = ActionValueBot(
        model_path=model_path,
        immediate_loss_penalty=immediate_loss_penalty,
    )
    model_policies: tuple[BotPolicy, ...] = (
        action_value_policy,
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
    return ActionValueEvaluationResult(
        model_vs_heuristic=model_summary,
        heuristic_only=heuristic_summary,
        random_only=random_summary,
    )


def _immediate_loss_delta(state: GameState, action_index: int) -> int:
    before = state.players[state.current_player_index].loss_score
    next_state = state
    for action in resolve_turn_action_before_refill(state, action_index):
        next_state = apply_known_legal_action(next_state, action)
    after = next_state.players[state.current_player_index].loss_score
    return after - before


def action_value_evaluation_result_to_dict(
    result: ActionValueEvaluationResult,
) -> dict[str, dict[str, object]]:
    """Convert evaluation summaries into JSON-friendly data."""
    return {
        "model_vs_heuristic": evaluation_summary_to_dict(result.model_vs_heuristic),
        "heuristic_only": evaluation_summary_to_dict(result.heuristic_only),
        "random_only": evaluation_summary_to_dict(result.random_only),
    }


def write_action_value_evaluation_result(
    result: ActionValueEvaluationResult,
    *,
    json_output: Path | None = None,
    csv_output: Path | None = None,
) -> None:
    """Write evaluation results as JSON and/or CSV."""
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(
                action_value_evaluation_result_to_dict(result),
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
                    "average_loss_shares",
                    "average_loss_scores",
                    "average_turn_count",
                    "total_elapsed_seconds",
                    "average_elapsed_seconds",
                ],
            )
            writer.writeheader()
            for scenario, summary in action_value_evaluation_result_to_dict(
                result
            ).items():
                writer.writerow({"scenario": scenario, **summary})


def _format_summary(name: str, summary: EvaluationSummary) -> str:
    return (
        f"{name}: matches={summary.match_count} "
        f"win_rates={summary.win_rates} "
        f"loss_share={summary.average_loss_shares} "
        f"avg_turns={summary.average_turn_count:.2f}"
    )


def _parse_args() -> tuple[Path, tuple[int, ...], float, Path | None, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--immediate-loss-penalty", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    return (
        args.model_path,
        seeds,
        args.immediate_loss_penalty,
        args.json_output,
        args.csv_output,
    )


def main() -> None:
    """Evaluate a greedy action-value policy from command-line arguments."""
    model_path, seeds, immediate_loss_penalty, json_output, csv_output = _parse_args()
    result = evaluate_action_value_model(
        model_path,
        seeds=seeds,
        immediate_loss_penalty=immediate_loss_penalty,
    )
    write_action_value_evaluation_result(
        result,
        json_output=json_output,
        csv_output=csv_output,
    )
    print(_format_summary("model_vs_heuristic", result.model_vs_heuristic))
    print(_format_summary("heuristic_only", result.heuristic_only))
    print(_format_summary("random_only", result.random_only))


if __name__ == "__main__":
    main()
