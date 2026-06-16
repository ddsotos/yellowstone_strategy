"""Diagnostics for turn-level learned policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action
from yellowstone.gym_env import YellowstoneTurnGymEnv, gymnasium_available
from yellowstone.turn_action_space import (
    TurnAction,
    legal_turn_action_indices,
    resolve_turn_action,
    turn_action_from_index,
    turn_action_to_index,
)
from yellowstone.types import GameState, Phase, PlaceCardAction

try:
    import numpy as np
    from sb3_contrib import MaskablePPO
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    MaskablePPO = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class PolicyDiagnostics:
    """Summary of learned turn choices on learner turns."""

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

    @property
    def heuristic_two_rate_when_no_damage_two_available(self) -> float:
        return _ratio(
            self.heuristic_two_when_no_damage_two_available,
            self.no_damage_two_available_turns,
        )


def run_policy_diagnostics(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    deterministic: bool = True,
) -> PolicyDiagnostics:
    """Run model-vs-heuristic games and summarize learner turn choices."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    _require_dependencies()

    model = MaskablePPO.load(model_path)
    model_path_text = str(model_path)
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
        env = YellowstoneTurnGymEnv(normalize_observations=True)
        observation, _ = env.reset(seed=seed)
        done = False
        truncated = False
        while not done and not truncated:
            state = env.env.state
            if state is None or state.phase == Phase.GAME_OVER:
                break

            mask = env.action_masks()
            action_index, _ = model.predict(
                observation,
                deterministic=deterministic,
                action_masks=mask,
            )
            model_action_index = int(action_index)
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

            observation, _, done, truncated, _ = env.step(model_action_index)

    return PolicyDiagnostics(
        model_path=model_path_text,
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


def write_policy_diagnostics(
    diagnostics: PolicyDiagnostics,
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
        "heuristic_two_rate_when_no_damage_two_available": (
            diagnostics.heuristic_two_rate_when_no_damage_two_available
        ),
    }
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _heuristic_turn_action_index(state: GameState) -> int:
    bot = HeuristicBot()
    low_level_actions = []
    current_state = state
    while current_state.current_player_index == state.current_player_index:
        action = bot.choose_action(current_state)
        if action is None:
            break
        low_level_actions.append(action)
        current_state = apply_known_legal_action(current_state, action)
        if current_state.phase == Phase.GAME_OVER:
            break
        if (
            current_state.phase == Phase.PLAY
            and current_state.cards_played_this_turn == 0
        ):
            break

    hand_indices = tuple(
        action.hand_index
        for action in low_level_actions
        if isinstance(action, PlaceCardAction)
    )
    if len(hand_indices) == 1:
        return turn_action_to_index(TurnAction((hand_indices[0],)))
    if len(hand_indices) >= 2:
        first_index = hand_indices[0]
        second_index = hand_indices[1]
        if second_index >= first_index:
            second_index += 1
        return turn_action_to_index(TurnAction((first_index, second_index)))
    return legal_turn_action_indices(state)[0]


def _has_no_damage_two_card_action(state: GameState) -> bool:
    before_count = len(state.players[state.current_player_index].negative_cards)
    for action_index in legal_turn_action_indices(state):
        turn_action = turn_action_from_index(action_index)
        if len(turn_action.hand_indices) != 2:
            continue
        current_state = state
        try:
            for low_level_action in resolve_turn_action(state, action_index):
                current_state = apply_known_legal_action(current_state, low_level_action)
        except ValueError:
            continue
        after_count = len(current_state.players[state.current_player_index].negative_cards)
        if after_count == before_count:
            return True
    return False


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _require_dependencies() -> None:
    if not (gymnasium_available() and np is not None and MaskablePPO is not None):
        raise ImportError(
            "RL model diagnostics dependencies are not installed. "
            'Install them with: python -m pip install -e ".[rl]"'
        )


def _parse_args() -> tuple[Path, tuple[int, ...], bool, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    return args.model_path, seeds, not args.stochastic, args.json_output


def main() -> None:
    """Run policy diagnostics from command-line arguments."""
    model_path, seeds, deterministic, json_output = _parse_args()
    diagnostics = run_policy_diagnostics(
        model_path,
        seeds=seeds,
        deterministic=deterministic,
    )
    write_policy_diagnostics(diagnostics, json_output=json_output)
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
