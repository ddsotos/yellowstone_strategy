"""Diagnostics for greedy action-value policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from yellowstone.action_value_model import predict_action_values
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.policy_diagnostics import (
    _has_no_damage_two_card_action,
    _heuristic_turn_action_index,
)
from yellowstone.turn_action_space import (
    legal_turn_action_indices,
    resolve_turn_action,
    resolve_turn_action_before_refill,
    turn_action_from_index,
)
from yellowstone.types import Phase


@dataclass(frozen=True, slots=True)
class ActionValueDiagnostics:
    """Summary of greedy action-value turn choices."""

    model_path: str
    games: int
    learner_turns: int
    model_one_card_turns: int
    model_two_card_turns: int
    heuristic_one_card_turns: int
    heuristic_two_card_turns: int
    same_action_turns: int
    no_damage_two_available_turns: int
    model_two_when_no_damage_two_available: int
    heuristic_two_when_no_damage_two_available: int

    @property
    def model_two_card_rate(self) -> float:
        return _ratio(self.model_two_card_turns, self.learner_turns)

    @property
    def heuristic_two_card_rate(self) -> float:
        return _ratio(self.heuristic_two_card_turns, self.learner_turns)

    @property
    def same_action_rate(self) -> float:
        return _ratio(self.same_action_turns, self.learner_turns)

    @property
    def model_two_rate_when_no_damage_two_available(self) -> float:
        return _ratio(
            self.model_two_when_no_damage_two_available,
            self.no_damage_two_available_turns,
        )


def run_action_value_diagnostics(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    immediate_loss_penalty: float = 0.0,
    max_actions: int = 10_000,
) -> ActionValueDiagnostics:
    """Run games and summarize greedy action-value choices."""
    learner_turns = 0
    model_one_card_turns = 0
    model_two_card_turns = 0
    heuristic_one_card_turns = 0
    heuristic_two_card_turns = 0
    same_action_turns = 0
    no_damage_two_available_turns = 0
    model_two_when_no_damage_two_available = 0
    heuristic_two_when_no_damage_two_available = 0

    for seed in seeds:
        state = create_initial_state(4, seed=seed)
        action_count = 0
        while state.phase != Phase.GAME_OVER and action_count < max_actions:
            if state.current_player_index != 0:
                action = _heuristic_low_level_action(state)
                if action is None:
                    break
                state = apply_known_legal_action(state, action)
                action_count += 1
                continue

            action_indices = legal_turn_action_indices(state)
            if not action_indices:
                break
            values = predict_action_values(
                state,
                action_indices=action_indices,
                model_path=model_path,
            )
            model_action_index, _ = min(
                (
                    (
                        action_index,
                        value
                        + immediate_loss_penalty
                        * _immediate_loss_delta(state, action_index),
                    )
                    for action_index, value in zip(action_indices, values, strict=True)
                ),
                key=lambda item: (item[1], item[0]),
            )
            heuristic_action_index = _heuristic_turn_action_index(state)
            model_action = turn_action_from_index(model_action_index)
            heuristic_action = turn_action_from_index(heuristic_action_index)
            no_damage_two_available = _has_no_damage_two_card_action(state)

            learner_turns += 1
            if len(model_action.hand_indices) == 1:
                model_one_card_turns += 1
            else:
                model_two_card_turns += 1
            if len(heuristic_action.hand_indices) == 1:
                heuristic_one_card_turns += 1
            else:
                heuristic_two_card_turns += 1
            if model_action_index == heuristic_action_index:
                same_action_turns += 1
            if no_damage_two_available:
                no_damage_two_available_turns += 1
                if len(model_action.hand_indices) == 2:
                    model_two_when_no_damage_two_available += 1
                if len(heuristic_action.hand_indices) == 2:
                    heuristic_two_when_no_damage_two_available += 1

            for action in resolve_turn_action(state, model_action_index):
                state = apply_known_legal_action(state, action)
                action_count += 1
                if state.phase == Phase.GAME_OVER:
                    break

    return ActionValueDiagnostics(
        model_path=str(model_path),
        games=len(seeds),
        learner_turns=learner_turns,
        model_one_card_turns=model_one_card_turns,
        model_two_card_turns=model_two_card_turns,
        heuristic_one_card_turns=heuristic_one_card_turns,
        heuristic_two_card_turns=heuristic_two_card_turns,
        same_action_turns=same_action_turns,
        no_damage_two_available_turns=no_damage_two_available_turns,
        model_two_when_no_damage_two_available=model_two_when_no_damage_two_available,
        heuristic_two_when_no_damage_two_available=(
            heuristic_two_when_no_damage_two_available
        ),
    )


def write_action_value_diagnostics(
    diagnostics: ActionValueDiagnostics,
    *,
    json_output: Path | None = None,
) -> None:
    """Write diagnostics as JSON."""
    if json_output is None:
        return
    json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(diagnostics) | {
        "model_two_card_rate": diagnostics.model_two_card_rate,
        "heuristic_two_card_rate": diagnostics.heuristic_two_card_rate,
        "same_action_rate": diagnostics.same_action_rate,
        "model_two_rate_when_no_damage_two_available": (
            diagnostics.model_two_rate_when_no_damage_two_available
        ),
    }
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _heuristic_low_level_action(state):
    from yellowstone.bots import HeuristicBot

    return HeuristicBot().choose_action(state)


def _immediate_loss_delta(state, action_index: int) -> int:
    before = state.players[state.current_player_index].loss_score
    next_state = state
    for action in resolve_turn_action_before_refill(state, action_index):
        next_state = apply_known_legal_action(next_state, action)
    after = next_state.players[state.current_player_index].loss_score
    return after - before


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_args() -> tuple[Path, tuple[int, ...], float, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--immediate-loss-penalty", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    return args.model_path, seeds, args.immediate_loss_penalty, args.json_output


def main() -> None:
    """Run action-value diagnostics from command-line arguments."""
    model_path, seeds, immediate_loss_penalty, json_output = _parse_args()
    diagnostics = run_action_value_diagnostics(
        model_path,
        seeds=seeds,
        immediate_loss_penalty=immediate_loss_penalty,
    )
    write_action_value_diagnostics(diagnostics, json_output=json_output)
    print(f"model_path={diagnostics.model_path}")
    print(f"games={diagnostics.games}")
    print(f"learner_turns={diagnostics.learner_turns}")
    print(f"model_two_card_rate={diagnostics.model_two_card_rate:.3f}")
    print(f"heuristic_two_card_rate={diagnostics.heuristic_two_card_rate:.3f}")
    print(f"same_action_rate={diagnostics.same_action_rate:.3f}")
    print(
        "model_two_rate_when_no_damage_two_available="
        f"{diagnostics.model_two_rate_when_no_damage_two_available:.3f}"
    )


if __name__ == "__main__":
    main()
