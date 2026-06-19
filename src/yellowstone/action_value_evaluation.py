"""Evaluate a supervised action-value model as a greedy policy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter

from yellowstone.action_value_dataset import _apply_rollout_action
from yellowstone.action_value_model import predict_action_values
from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.evaluation import (
    EvaluationSummary,
    evaluate_policies,
    make_heuristic_policies,
    make_random_policies,
)
from yellowstone.model_evaluation import evaluation_summary_to_dict
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.policy_diagnostics import _heuristic_turn_action_index
from yellowstone.turn_action_space import (
    legal_turn_action_indices,
    resolve_turn_action,
    resolve_turn_action_before_refill,
)
from yellowstone.types import Action, GameState, Phase


@dataclass(frozen=True, slots=True)
class ActionValueEvaluationResult:
    """Evaluation summaries for a greedy action-value policy."""

    model_vs_heuristic: EvaluationSummary
    heuristic_only: EvaluationSummary
    random_only: EvaluationSummary


@dataclass(frozen=True, slots=True)
class ContinuingEvaluationSummary:
    """Aggregate continuing-game loss-share metrics."""

    match_count: int
    learner_turns: int
    average_loss_deltas: tuple[float, ...]
    average_loss_shares: tuple[float, ...]
    average_action_count: float
    total_elapsed_seconds: float
    average_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ContinuingActionValueEvaluationResult:
    """Continuing-game summaries for action-value and baseline policies."""

    model_vs_heuristic: ContinuingEvaluationSummary
    heuristic_only: ContinuingEvaluationSummary


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


@dataclass(slots=True)
class AdvantageGatedActionValueBot:
    """Use an advantage model only when it beats heuristic by a margin."""

    model_path: Path
    advantage_margin: float = 0.5
    loss_guard: int = 2
    pending_actions: tuple[Action, ...] = ()

    def choose_action(self, state: GameState) -> Action | None:
        """Choose heuristic unless a candidate has enough predicted advantage."""
        if self.pending_actions:
            action = self.pending_actions[0]
            self.pending_actions = self.pending_actions[1:]
            return action

        action_indices = legal_turn_action_indices(state)
        if not action_indices:
            return None
        heuristic_action_index = _heuristic_turn_action_index(state)
        if heuristic_action_index not in action_indices:
            heuristic_action_index = action_indices[0]
        values = predict_action_values(
            state,
            action_indices=action_indices,
            model_path=self.model_path,
        )
        best_action_index, best_advantage = max(
            zip(action_indices, values, strict=True),
            key=lambda item: (item[1], -item[0]),
        )
        selected_action_index = heuristic_action_index
        if best_advantage > self.advantage_margin:
            heuristic_loss_delta = _immediate_loss_delta(state, heuristic_action_index)
            best_loss_delta = _immediate_loss_delta(state, best_action_index)
            if best_loss_delta - heuristic_loss_delta <= self.loss_guard:
                selected_action_index = best_action_index

        actions = resolve_turn_action(state, selected_action_index)
        if not actions:
            return None
        self.pending_actions = actions[1:]
        return actions[0]


def evaluate_action_value_model(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    immediate_loss_penalty: float = 0.0,
    policy: str = "greedy_q",
    advantage_margin: float = 0.5,
    loss_guard: int = 2,
) -> ActionValueEvaluationResult:
    """Evaluate a greedy action-value policy against heuristic NPCs."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    action_value_policy = _make_action_value_policy(
        model_path,
        policy=policy,
        immediate_loss_penalty=immediate_loss_penalty,
        advantage_margin=advantage_margin,
        loss_guard=loss_guard,
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


def evaluate_action_value_model_continuing(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    learner_turns: int,
    immediate_loss_penalty: float = 0.0,
    policy: str = "greedy_q",
    advantage_margin: float = 0.5,
    loss_guard: int = 2,
    max_actions: int = 100_000,
) -> ContinuingActionValueEvaluationResult:
    """Evaluate a greedy action-value policy in continuing-game mode."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    if learner_turns <= 0:
        raise ValueError("learner_turns must be positive")
    model_results = tuple(
        _run_continuing_match(
            (
                _make_action_value_policy(
                    model_path,
                    policy=policy,
                    immediate_loss_penalty=immediate_loss_penalty,
                    advantage_margin=advantage_margin,
                    loss_guard=loss_guard,
                ),
                HeuristicBot(),
                HeuristicBot(),
                HeuristicBot(),
            ),
            seed=seed,
            learner_turns=learner_turns,
            max_actions=max_actions,
        )
        for seed in seeds
    )
    heuristic_results = tuple(
        _run_continuing_match(
            (HeuristicBot(), HeuristicBot(), HeuristicBot(), HeuristicBot()),
            seed=seed,
            learner_turns=learner_turns,
            max_actions=max_actions,
        )
        for seed in seeds
    )
    return ContinuingActionValueEvaluationResult(
        model_vs_heuristic=_summarize_continuing_results(
            model_results,
            player_count=4,
            learner_turns=learner_turns,
        ),
        heuristic_only=_summarize_continuing_results(
            heuristic_results,
            player_count=4,
            learner_turns=learner_turns,
        ),
    )


def _make_action_value_policy(
    model_path: Path,
    *,
    policy: str,
    immediate_loss_penalty: float,
    advantage_margin: float,
    loss_guard: int,
) -> BotPolicy:
    if policy == "greedy_q":
        return ActionValueBot(
            model_path=model_path,
            immediate_loss_penalty=immediate_loss_penalty,
        )
    if policy == "advantage_gated":
        return AdvantageGatedActionValueBot(
            model_path=model_path,
            advantage_margin=advantage_margin,
            loss_guard=loss_guard,
        )
    raise ValueError(f"unsupported policy: {policy}")


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


def continuing_action_value_evaluation_result_to_dict(
    result: ContinuingActionValueEvaluationResult,
) -> dict[str, dict[str, object]]:
    """Convert continuing-game summaries into JSON-friendly data."""
    return {
        "model_vs_heuristic": _continuing_summary_to_dict(
            result.model_vs_heuristic
        ),
        "heuristic_only": _continuing_summary_to_dict(result.heuristic_only),
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


def write_continuing_action_value_evaluation_result(
    result: ContinuingActionValueEvaluationResult,
    *,
    json_output: Path | None = None,
    csv_output: Path | None = None,
) -> None:
    """Write continuing-game evaluation results as JSON and/or CSV."""
    result_dict = continuing_action_value_evaluation_result_to_dict(result)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2) + "\n",
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
                    "learner_turns",
                    "average_loss_deltas",
                    "average_loss_shares",
                    "average_action_count",
                    "total_elapsed_seconds",
                    "average_elapsed_seconds",
                ],
            )
            writer.writeheader()
            for scenario, summary in result_dict.items():
                writer.writerow({"scenario": scenario, **summary})


def _format_summary(name: str, summary: EvaluationSummary) -> str:
    return (
        f"{name}: matches={summary.match_count} "
        f"win_rates={summary.win_rates} "
        f"loss_share={summary.average_loss_shares} "
        f"avg_turns={summary.average_turn_count:.2f}"
    )


def _format_continuing_summary(
    name: str,
    summary: ContinuingEvaluationSummary,
) -> str:
    return (
        f"{name}: matches={summary.match_count} "
        f"learner_turns={summary.learner_turns} "
        f"loss_delta={summary.average_loss_deltas} "
        f"loss_share={summary.average_loss_shares} "
        f"avg_actions={summary.average_action_count:.2f}"
    )


@dataclass(frozen=True, slots=True)
class _ContinuingMatchResult:
    loss_deltas: tuple[int, ...]
    action_count: int
    elapsed_seconds: float


def _run_continuing_match(
    policies: tuple[BotPolicy, ...],
    *,
    seed: int,
    learner_turns: int,
    max_actions: int,
) -> _ContinuingMatchResult:
    state = create_initial_state(len(policies), seed=seed)
    baseline_losses = tuple(player.loss_score for player in state.players)
    rng = Random(seed)
    action_count = 0
    seen_learner_turns = 0
    started_at = perf_counter()

    while action_count < max_actions:
        if (
            state.phase == Phase.PLAY
            and state.current_player_index == 0
            and state.cards_played_this_turn == 0
        ):
            if seen_learner_turns >= learner_turns:
                break
            seen_learner_turns += 1
        action = policies[state.current_player_index].choose_action(state)
        if action is None:
            break
        state = _apply_rollout_action(
            state,
            action,
            rng=rng,
            continuing_game=True,
        )
        action_count += 1

    elapsed_seconds = perf_counter() - started_at
    return _ContinuingMatchResult(
        loss_deltas=tuple(
            player.loss_score - baseline_loss
            for player, baseline_loss in zip(
                state.players,
                baseline_losses,
                strict=True,
            )
        ),
        action_count=action_count,
        elapsed_seconds=elapsed_seconds,
    )


def _summarize_continuing_results(
    results: tuple[_ContinuingMatchResult, ...],
    *,
    player_count: int,
    learner_turns: int,
) -> ContinuingEvaluationSummary:
    if not results:
        return ContinuingEvaluationSummary(
            match_count=0,
            learner_turns=learner_turns,
            average_loss_deltas=tuple(0.0 for _ in range(player_count)),
            average_loss_shares=tuple(0.0 for _ in range(player_count)),
            average_action_count=0.0,
            total_elapsed_seconds=0.0,
            average_elapsed_seconds=0.0,
        )
    match_count = len(results)
    total_elapsed = sum(result.elapsed_seconds for result in results)
    return ContinuingEvaluationSummary(
        match_count=match_count,
        learner_turns=learner_turns,
        average_loss_deltas=tuple(
            sum(result.loss_deltas[player_index] for result in results) / match_count
            for player_index in range(player_count)
        ),
        average_loss_shares=tuple(
            sum(_loss_delta_share(result.loss_deltas, player_index) for result in results)
            / match_count
            for player_index in range(player_count)
        ),
        average_action_count=sum(result.action_count for result in results)
        / match_count,
        total_elapsed_seconds=total_elapsed,
        average_elapsed_seconds=total_elapsed / match_count,
    )


def _loss_delta_share(loss_deltas: tuple[int, ...], player_index: int) -> float:
    total_loss_delta = sum(loss_deltas)
    if total_loss_delta <= 0:
        return 0.0
    return loss_deltas[player_index] / total_loss_delta


def _continuing_summary_to_dict(
    summary: ContinuingEvaluationSummary,
) -> dict[str, object]:
    return asdict(summary)


@dataclass(frozen=True, slots=True)
class _ParsedArgs:
    model_path: Path
    seeds: tuple[int, ...]
    immediate_loss_penalty: float
    json_output: Path | None
    csv_output: Path | None
    continuing_learner_turns: int | None
    max_actions: int
    policy: str
    advantage_margin: float
    loss_guard: int


def _parse_args() -> _ParsedArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--immediate-loss-penalty", type=float, default=0.0)
    parser.add_argument(
        "--policy",
        choices=("greedy_q", "advantage_gated"),
        default="greedy_q",
    )
    parser.add_argument("--advantage-margin", type=float, default=0.5)
    parser.add_argument("--loss-guard", type=int, default=2)
    parser.add_argument("--continuing-learner-turns", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=100_000)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    return _ParsedArgs(
        model_path=args.model_path,
        seeds=seeds,
        immediate_loss_penalty=args.immediate_loss_penalty,
        json_output=args.json_output,
        csv_output=args.csv_output,
        continuing_learner_turns=args.continuing_learner_turns,
        max_actions=args.max_actions,
        policy=args.policy,
        advantage_margin=args.advantage_margin,
        loss_guard=args.loss_guard,
    )


def main() -> None:
    """Evaluate a greedy action-value policy from command-line arguments."""
    args = _parse_args()
    if args.continuing_learner_turns is not None:
        result = evaluate_action_value_model_continuing(
            args.model_path,
            seeds=args.seeds,
            learner_turns=args.continuing_learner_turns,
            immediate_loss_penalty=args.immediate_loss_penalty,
            policy=args.policy,
            advantage_margin=args.advantage_margin,
            loss_guard=args.loss_guard,
            max_actions=args.max_actions,
        )
        write_continuing_action_value_evaluation_result(
            result,
            json_output=args.json_output,
            csv_output=args.csv_output,
        )
        print(
            _format_continuing_summary(
                "model_vs_heuristic",
                result.model_vs_heuristic,
            )
        )
        print(
            _format_continuing_summary("heuristic_only", result.heuristic_only)
        )
        return
    result = evaluate_action_value_model(
        args.model_path,
        seeds=args.seeds,
        immediate_loss_penalty=args.immediate_loss_penalty,
        policy=args.policy,
        advantage_margin=args.advantage_margin,
        loss_guard=args.loss_guard,
    )
    write_action_value_evaluation_result(
        result,
        json_output=args.json_output,
        csv_output=args.csv_output,
    )
    print(_format_summary("model_vs_heuristic", result.model_vs_heuristic))
    print(_format_summary("heuristic_only", result.heuristic_only))
    print(_format_summary("random_only", result.random_only))


if __name__ == "__main__":
    main()
