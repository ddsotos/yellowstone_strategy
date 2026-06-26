"""Collect supervised samples for binary one-card/two-card decisions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from yellowstone.action_value_dataset import _apply_rollout_action
from yellowstone.bots import HeuristicBot
from yellowstone.game import create_initial_state
from yellowstone.heuristic_turn_plan import (
    choose_heuristic_one_card_plan,
    choose_heuristic_two_card_plan,
)
from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.two_card_decision import (
    TwoCardDecision,
    TwoCardDecisionCase,
    choose_two_card_decision_by_rollout,
    classify_two_card_decision_point_with_plan,
)
from yellowstone.types import GameState, Phase


@dataclass(frozen=True, slots=True)
class TwoCardDecisionSample:
    """One turn-start observation labeled with a binary rollout decision."""

    observation: tuple[float, ...]
    selected_decision: int
    stop_self_loss: float
    stop_relative_loss: float
    two_card_self_loss: float
    two_card_relative_loss: float
    decision_case: str
    player_index: int
    hand_count: int
    seed: int
    turn_index: int


@dataclass(frozen=True, slots=True)
class TwoCardDecisionDatasetSummary:
    """Summary of generated binary-decision samples."""

    source_games: int
    source_states: int
    samples: int
    horizon_learner_turns: int
    hand_count_histogram: tuple[int, ...]
    selected_decision_histogram: tuple[int, int]
    decision_case_histogram: dict[str, int]
    source_seed_count: int
    rollout_count: int = 1
    continuing_game: bool = False


def collect_two_card_decision_samples(
    *,
    source_games: int,
    source_seed_start: int = 1,
    source_state_limit: int = 1000,
    horizon_learner_turns: int = 4,
    learning_player_index: int = 0,
    player_count: int = 4,
    max_actions: int = 10_000,
    continuing_game: bool = False,
    target: str = "relative_loss",
    max_source_states_per_game: int | None = None,
    rollout_count: int = 1,
) -> tuple[TwoCardDecisionSample, ...]:
    """Collect turn-start samples labeled by binary rollout comparison."""
    if max_source_states_per_game is not None and max_source_states_per_game <= 0:
        raise ValueError("max_source_states_per_game must be positive")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    source_states = _collect_source_states(
        source_games=source_games,
        source_seed_start=source_seed_start,
        source_state_limit=source_state_limit,
        learning_player_index=learning_player_index,
        player_count=player_count,
        max_actions=max_actions,
        continuing_game=continuing_game,
        max_source_states_per_game=max_source_states_per_game,
    )
    samples: list[TwoCardDecisionSample] = []
    for turn_index, (state, source_seed) in enumerate(source_states):
        one_card_plan = choose_heuristic_one_card_plan(state)
        two_card_plan = choose_heuristic_two_card_plan(state)
        if one_card_plan is None or two_card_plan is None:
            continue
        decision_case = classify_two_card_decision_point_with_plan(
            state, two_card_plan
        )
        if decision_case == TwoCardDecisionCase.NOT_APPLICABLE:
            continue
        try:
            result = choose_two_card_decision_by_rollout(
                state,
                learning_player_index=learning_player_index,
                horizon_learner_turns=horizon_learner_turns,
                seed=source_seed_start + turn_index,
                max_actions=max_actions,
                continuing_game=continuing_game,
                target=target,
                plans=(one_card_plan, two_card_plan),
                rollout_count=rollout_count,
            )
        except ValueError:
            continue
        values = {value.decision: value for value in result.values}
        stop_value = values[TwoCardDecision.STOP_AFTER_ONE]
        two_card_value = values[TwoCardDecision.PLAY_SECOND_HEURISTIC]
        player = state.players[learning_player_index]
        samples.append(
            TwoCardDecisionSample(
                observation=normalize_observation(
                    state_to_observation(
                        state,
                        heuristic_bonuses=(
                            one_card_plan.bonus_score,
                            two_card_plan.bonus_score,
                        ),
                        heuristic_negative_deltas=(
                            one_card_plan.negative_card_delta,
                            two_card_plan.negative_card_delta,
                        ),
                    )
                ),
                selected_decision=_decision_to_label(result.selected_decision),
                stop_self_loss=stop_value.target_self_loss,
                stop_relative_loss=stop_value.target_relative_loss,
                two_card_self_loss=two_card_value.target_self_loss,
                two_card_relative_loss=two_card_value.target_relative_loss,
                decision_case=decision_case.value,
                player_index=learning_player_index,
                hand_count=len(player.hand),
                seed=source_seed,
                turn_index=turn_index,
            )
        )
    return tuple(samples)


def summarize_two_card_decision_samples(
    samples: Iterable[TwoCardDecisionSample],
    *,
    source_games: int,
    source_states: int,
    horizon_learner_turns: int,
    continuing_game: bool = False,
    rollout_count: int = 1,
) -> TwoCardDecisionDatasetSummary:
    """Summarize sample coverage for binary decision data."""
    sample_tuple = tuple(samples)
    hand_histogram = [0] * 7
    selected_histogram = [0, 0]
    case_histogram: dict[str, int] = {}
    for sample in sample_tuple:
        if 0 <= sample.hand_count < len(hand_histogram):
            hand_histogram[sample.hand_count] += 1
        if 0 <= sample.selected_decision <= 1:
            selected_histogram[sample.selected_decision] += 1
        case_histogram[sample.decision_case] = (
            case_histogram.get(sample.decision_case, 0) + 1
        )
    return TwoCardDecisionDatasetSummary(
        source_games=source_games,
        source_states=source_states,
        samples=len(sample_tuple),
        horizon_learner_turns=horizon_learner_turns,
        hand_count_histogram=tuple(hand_histogram),
        selected_decision_histogram=(
            selected_histogram[0],
            selected_histogram[1],
        ),
        decision_case_histogram=case_histogram,
        source_seed_count=len({sample.seed for sample in sample_tuple}),
        rollout_count=rollout_count,
        continuing_game=continuing_game,
    )


def write_two_card_decision_samples(
    samples: Iterable[TwoCardDecisionSample],
    path: str | Path,
) -> None:
    """Write samples as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(_sample_to_dict(sample), separators=(",", ":")))
            file.write("\n")


def read_two_card_decision_samples(
    path: str | Path,
) -> tuple[TwoCardDecisionSample, ...]:
    """Read binary decision samples from JSON Lines."""
    samples: list[TwoCardDecisionSample] = []
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                samples.append(_sample_from_dict(json.loads(line)))
    return tuple(samples)


def two_card_decision_dataset_summary_to_dict(
    summary: TwoCardDecisionDatasetSummary,
) -> dict[str, int | bool | list[int] | dict[str, int]]:
    """Convert a summary to JSON-friendly data."""
    return {
        "source_games": summary.source_games,
        "source_states": summary.source_states,
        "samples": summary.samples,
        "horizon_learner_turns": summary.horizon_learner_turns,
        "hand_count_histogram": list(summary.hand_count_histogram),
        "selected_decision_histogram": list(summary.selected_decision_histogram),
        "decision_case_histogram": dict(summary.decision_case_histogram),
        "source_seed_count": summary.source_seed_count,
        "rollout_count": summary.rollout_count,
        "continuing_game": summary.continuing_game,
    }


def _collect_source_states(
    *,
    source_games: int,
    source_seed_start: int,
    source_state_limit: int,
    learning_player_index: int,
    player_count: int,
    max_actions: int,
    continuing_game: bool,
    max_source_states_per_game: int | None,
) -> tuple[tuple[GameState, int], ...]:
    source_states: list[tuple[GameState, int]] = []
    bot = HeuristicBot()
    for offset in range(source_games):
        if len(source_states) >= source_state_limit:
            break
        seed = source_seed_start + offset
        rng = Random(seed)
        state = create_initial_state(player_count, seed=seed)
        action_count = 0
        source_states_from_game = 0
        while (
            (continuing_game or state.phase != Phase.GAME_OVER)
            and action_count < max_actions
        ):
            if _is_learner_turn_start(state, learning_player_index):
                source_states.append((state, seed))
                source_states_from_game += 1
                if len(source_states) >= source_state_limit:
                    break
                if (
                    max_source_states_per_game is not None
                    and source_states_from_game >= max_source_states_per_game
                ):
                    break
            action = bot.choose_action(state)
            if action is None:
                break
            state = _apply_rollout_action(
                state,
                action,
                rng=rng,
                continuing_game=continuing_game,
            )
            action_count += 1
    return tuple(source_states)


def _is_learner_turn_start(state: GameState, learning_player_index: int) -> bool:
    return (
        state.phase == Phase.PLAY
        and state.current_player_index == learning_player_index
        and state.cards_played_this_turn == 0
    )


def _decision_to_label(decision: TwoCardDecision) -> int:
    if decision == TwoCardDecision.STOP_AFTER_ONE:
        return 0
    if decision == TwoCardDecision.PLAY_SECOND_HEURISTIC:
        return 1
    raise ValueError(f"unsupported decision: {decision}")


def _sample_to_dict(sample: TwoCardDecisionSample) -> dict[str, object]:
    return {
        "observation": list(sample.observation),
        "selected_decision": sample.selected_decision,
        "stop_self_loss": sample.stop_self_loss,
        "stop_relative_loss": sample.stop_relative_loss,
        "two_card_self_loss": sample.two_card_self_loss,
        "two_card_relative_loss": sample.two_card_relative_loss,
        "decision_case": sample.decision_case,
        "player_index": sample.player_index,
        "hand_count": sample.hand_count,
        "seed": sample.seed,
        "turn_index": sample.turn_index,
    }


def _sample_from_dict(data: dict[str, object]) -> TwoCardDecisionSample:
    observation = tuple(float(value) for value in data["observation"])  # type: ignore[index]
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_SIZE}")
    return TwoCardDecisionSample(
        observation=observation,
        selected_decision=int(data["selected_decision"]),
        stop_self_loss=float(data["stop_self_loss"]),
        stop_relative_loss=float(data["stop_relative_loss"]),
        two_card_self_loss=float(data["two_card_self_loss"]),
        two_card_relative_loss=float(data["two_card_relative_loss"]),
        decision_case=str(data["decision_case"]),
        player_index=int(data["player_index"]),
        hand_count=int(data["hand_count"]),
        seed=int(data["seed"]),
        turn_index=int(data["turn_index"]),
    )


def main() -> None:
    """CLI entry point for binary decision data collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-games", type=int, default=100)
    parser.add_argument("--source-seed-start", type=int, default=1)
    parser.add_argument("--source-state-limit", type=int, default=1000)
    parser.add_argument("--horizon-learner-turns", type=int, default=4)
    parser.add_argument("--continuing-game", action="store_true")
    parser.add_argument("--max-source-states-per-game", type=int)
    parser.add_argument("--rollout-count", type=int, default=1)
    parser.add_argument(
        "--target",
        choices=("self_loss", "relative_loss"),
        default="relative_loss",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    samples = collect_two_card_decision_samples(
        source_games=args.source_games,
        source_seed_start=args.source_seed_start,
        source_state_limit=args.source_state_limit,
        horizon_learner_turns=args.horizon_learner_turns,
        continuing_game=args.continuing_game,
        target=args.target,
        max_source_states_per_game=args.max_source_states_per_game,
        rollout_count=args.rollout_count,
    )
    write_two_card_decision_samples(samples, args.output)
    summary = summarize_two_card_decision_samples(
        samples,
        source_games=args.source_games,
        source_states=len({sample.turn_index for sample in samples}),
        horizon_learner_turns=args.horizon_learner_turns,
        continuing_game=args.continuing_game,
        rollout_count=args.rollout_count,
    )
    summary_dict = two_card_decision_dataset_summary_to_dict(summary)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
