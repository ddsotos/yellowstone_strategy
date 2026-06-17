"""Collect action-value samples for turn-level learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from yellowstone.bots import BotPolicy, ExploratoryHeuristicBot, HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.turn_action_space import (
    TURN_ACTION_SPACE_SIZE,
    legal_turn_action_indices,
    resolve_turn_action_before_refill,
)
from yellowstone.types import GameState, Phase


@dataclass(frozen=True, slots=True)
class ActionValueSample:
    """One turn-start action labeled by finite-horizon future loss."""

    observation: tuple[float, ...]
    action_index: int
    target_self_loss: float
    target_relative_loss: float
    player_index: int
    hand_count: int
    seed: int
    turn_index: int


@dataclass(frozen=True, slots=True)
class ActionValueDatasetSummary:
    """Summary of generated action-value samples."""

    source_games: int
    source_states: int
    samples: int
    horizon_learner_turns: int
    hand_count_histogram: tuple[int, ...]
    action_count_histogram: tuple[int, ...]


def collect_action_value_samples(
    *,
    source_games: int,
    source_seed_start: int = 1,
    source_state_limit: int = 1000,
    actions_per_state: int = 6,
    horizon_learner_turns: int = 20,
    exploratory_sources: bool = False,
    exploratory_one_card_probabilities: dict[int, float] | None = None,
    learning_player_index: int = 0,
    player_count: int = 4,
    max_actions: int = 10_000,
) -> tuple[ActionValueSample, ...]:
    """Collect finite-horizon Q(s,a)-style samples."""
    rng = Random(source_seed_start)
    source_states = _collect_source_states(
        source_games=source_games,
        source_seed_start=source_seed_start,
        source_state_limit=source_state_limit,
        exploratory_sources=exploratory_sources,
        exploratory_one_card_probabilities=exploratory_one_card_probabilities,
        learning_player_index=learning_player_index,
        player_count=player_count,
        max_actions=max_actions,
    )
    samples: list[ActionValueSample] = []
    for source_index, state in enumerate(source_states):
        player = state.players[learning_player_index]
        observation = normalize_observation(state_to_observation(state))
        for action_index in _sample_action_indices(
            state,
            actions_per_state=actions_per_state,
            rng=rng,
        ):
            after_decision = _apply_turn_action_before_refill(
                state,
                action_index=action_index,
                rng=rng,
            )
            result = _rollout_horizon(
                after_decision,
                learning_player_index=learning_player_index,
                horizon_learner_turns=horizon_learner_turns,
                rng=rng,
                max_actions=max_actions,
            )
            if result is None:
                continue
            target_self_loss, target_relative_loss = result
            samples.append(
                ActionValueSample(
                    observation=observation,
                    action_index=action_index,
                    target_self_loss=target_self_loss,
                    target_relative_loss=target_relative_loss,
                    player_index=learning_player_index,
                    hand_count=len(player.hand),
                    seed=source_seed_start + source_index,
                    turn_index=source_index,
                )
            )
    return tuple(samples)


def summarize_action_value_samples(
    samples: Iterable[ActionValueSample],
    *,
    source_games: int,
    source_states: int,
    horizon_learner_turns: int,
) -> ActionValueDatasetSummary:
    """Summarize sample and coverage counts."""
    sample_tuple = tuple(samples)
    hand_histogram = [0] * 7
    action_histogram = [0] * TURN_ACTION_SPACE_SIZE
    for sample in sample_tuple:
        if 0 <= sample.hand_count < len(hand_histogram):
            hand_histogram[sample.hand_count] += 1
        action_histogram[sample.action_index] += 1
    return ActionValueDatasetSummary(
        source_games=source_games,
        source_states=source_states,
        samples=len(sample_tuple),
        horizon_learner_turns=horizon_learner_turns,
        hand_count_histogram=tuple(hand_histogram),
        action_count_histogram=tuple(action_histogram),
    )


def write_action_value_samples(
    samples: Iterable[ActionValueSample],
    path: str | Path,
) -> None:
    """Write action-value samples as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(_sample_to_dict(sample), separators=(",", ":")))
            file.write("\n")


def read_action_value_samples(path: str | Path) -> tuple[ActionValueSample, ...]:
    """Read action-value samples from JSON Lines."""
    samples: list[ActionValueSample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(_sample_from_dict(json.loads(line)))
    return tuple(samples)


def action_value_dataset_summary_to_dict(
    summary: ActionValueDatasetSummary,
) -> dict[str, int | list[int]]:
    """Convert a summary to JSON-friendly data."""
    return {
        "source_games": summary.source_games,
        "source_states": summary.source_states,
        "samples": summary.samples,
        "horizon_learner_turns": summary.horizon_learner_turns,
        "hand_count_histogram": list(summary.hand_count_histogram),
        "action_count_histogram": list(summary.action_count_histogram),
    }


def _collect_source_states(
    *,
    source_games: int,
    source_seed_start: int,
    source_state_limit: int,
    exploratory_sources: bool,
    exploratory_one_card_probabilities: dict[int, float] | None,
    learning_player_index: int,
    player_count: int,
    max_actions: int,
) -> tuple[GameState, ...]:
    source_states: list[GameState] = []
    for offset in range(source_games):
        if len(source_states) >= source_state_limit:
            break
        seed = source_seed_start + offset
        rng = Random(seed)
        state = create_initial_state(player_count, seed=seed)
        policies = _make_policies(
            player_count=player_count,
            exploratory=exploratory_sources,
            seed=seed,
            one_card_probabilities=exploratory_one_card_probabilities,
        )
        action_count = 0
        while state.phase != Phase.GAME_OVER and action_count < max_actions:
            if _is_learner_turn_start(state, learning_player_index):
                source_states.append(state)
                if len(source_states) >= source_state_limit:
                    break
            action = policies[state.current_player_index].choose_action(state)
            if action is None:
                break
            state = apply_known_legal_action(state, action, rng=rng)
            action_count += 1
    return tuple(source_states)


def _make_policies(
    *,
    player_count: int,
    exploratory: bool,
    seed: int,
    one_card_probabilities: dict[int, float] | None,
) -> tuple[BotPolicy, ...]:
    if exploratory:
        return tuple(
            ExploratoryHeuristicBot(
                rng=Random(seed + index),
                one_card_probabilities=(
                    dict(one_card_probabilities)
                    if one_card_probabilities is not None
                    else {
                        6: 0.35,
                        5: 0.25,
                        4: 0.15,
                    }
                ),
            )
            for index in range(player_count)
        )
    return tuple(HeuristicBot() for _ in range(player_count))


def _sample_action_indices(
    state: GameState,
    *,
    actions_per_state: int,
    rng: Random,
) -> tuple[int, ...]:
    legal_indices = list(legal_turn_action_indices(state))
    if actions_per_state <= 0 or actions_per_state >= len(legal_indices):
        return tuple(legal_indices)
    rng.shuffle(legal_indices)
    return tuple(sorted(legal_indices[:actions_per_state]))


def _apply_turn_action_before_refill(
    state: GameState,
    *,
    action_index: int,
    rng: Random,
) -> GameState:
    next_state = state
    for action in resolve_turn_action_before_refill(next_state, action_index):
        next_state = apply_known_legal_action(next_state, action, rng=rng)
        if next_state.phase == Phase.GAME_OVER:
            break
    return next_state


def _rollout_horizon(
    state: GameState,
    *,
    learning_player_index: int,
    horizon_learner_turns: int,
    rng: Random,
    max_actions: int,
) -> tuple[float, float] | None:
    start_losses = tuple(player.loss_score for player in state.players)
    next_state = state
    bot = HeuristicBot()
    action_count = 0
    learner_turns = 0
    while (
        next_state.phase != Phase.GAME_OVER
        and action_count < max_actions
        and learner_turns < horizon_learner_turns
    ):
        if _is_learner_turn_start(next_state, learning_player_index):
            learner_turns += 1
            if learner_turns >= horizon_learner_turns:
                break
        action = bot.choose_action(next_state)
        if action is None:
            break
        next_state = apply_known_legal_action(next_state, action, rng=rng)
        action_count += 1
    if learner_turns < horizon_learner_turns:
        return None
    loss_deltas = tuple(
        player.loss_score - start_loss
        for player, start_loss in zip(next_state.players, start_losses, strict=True)
    )
    self_loss = float(loss_deltas[learning_player_index])
    average_loss = sum(loss_deltas) / len(loss_deltas)
    return self_loss, self_loss - average_loss


def _is_learner_turn_start(state: GameState, learning_player_index: int) -> bool:
    return (
        state.phase == Phase.PLAY
        and state.current_player_index == learning_player_index
        and state.cards_played_this_turn == 0
    )


def _sample_to_dict(sample: ActionValueSample) -> dict[str, object]:
    return {
        "observation": list(sample.observation),
        "action_index": sample.action_index,
        "target_self_loss": sample.target_self_loss,
        "target_relative_loss": sample.target_relative_loss,
        "player_index": sample.player_index,
        "hand_count": sample.hand_count,
        "seed": sample.seed,
        "turn_index": sample.turn_index,
    }


def _sample_from_dict(data: dict[str, object]) -> ActionValueSample:
    observation = tuple(float(value) for value in data["observation"])  # type: ignore[index]
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_SIZE}")
    action_index = int(data["action_index"])
    if not 0 <= action_index < TURN_ACTION_SPACE_SIZE:
        raise ValueError(f"action index out of range: {action_index}")
    return ActionValueSample(
        observation=observation,
        action_index=action_index,
        target_self_loss=float(data["target_self_loss"]),
        target_relative_loss=float(data["target_relative_loss"]),
        player_index=int(data["player_index"]),
        hand_count=int(data["hand_count"]),
        seed=int(data["seed"]),
        turn_index=int(data["turn_index"]),
    )


def _parse_one_card_probabilities(
    values: tuple[str, ...],
) -> dict[int, float] | None:
    if not values:
        return None
    probabilities: dict[int, float] = {}
    for value in values:
        hand_count_text, probability_text = value.split("=", maxsplit=1)
        hand_count = int(hand_count_text)
        probability = float(probability_text)
        if not 0 <= hand_count <= 6:
            raise ValueError(f"hand count must be 0..6: {hand_count}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be 0..1: {probability}")
        probabilities[hand_count] = probability
    return probabilities


def main() -> None:
    """CLI entry point for action-value data collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-games", type=int, default=100)
    parser.add_argument("--source-seed-start", type=int, default=1)
    parser.add_argument("--source-state-limit", type=int, default=1000)
    parser.add_argument("--actions-per-state", type=int, default=6)
    parser.add_argument("--horizon-learner-turns", type=int, default=20)
    parser.add_argument("--exploratory-sources", action="store_true")
    parser.add_argument(
        "--one-card-probability",
        action="append",
        metavar="HAND=PROBABILITY",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    samples = collect_action_value_samples(
        source_games=args.source_games,
        source_seed_start=args.source_seed_start,
        source_state_limit=args.source_state_limit,
        actions_per_state=args.actions_per_state,
        horizon_learner_turns=args.horizon_learner_turns,
        exploratory_sources=args.exploratory_sources,
        exploratory_one_card_probabilities=_parse_one_card_probabilities(
            tuple(args.one_card_probability or ())
        ),
    )
    write_action_value_samples(samples, args.output)
    summary = summarize_action_value_samples(
        samples,
        source_games=args.source_games,
        source_states=len({sample.turn_index for sample in samples}),
        horizon_learner_turns=args.horizon_learner_turns,
    )
    summary_dict = action_value_dataset_summary_to_dict(summary)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
