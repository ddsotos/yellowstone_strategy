"""Collect state-value samples from counterfactual learner turn actions."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Literal

from yellowstone.bots import BotPolicy, ExploratoryHeuristicBot, HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.observation import state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.turn_action_space import (
    legal_turn_action_indices,
    resolve_turn_action,
)
from yellowstone.types import GameState, Phase
from yellowstone.value_dataset import (
    StateValueSample,
    summarize_state_value_samples,
    state_value_dataset_summary_to_dict,
    write_state_value_samples,
)

CounterfactualTargetState = Literal["next-learner-turn", "after-refill"]


def collect_counterfactual_state_value_samples(
    *,
    source_games: int,
    source_seed_start: int = 1,
    source_state_limit: int = 1000,
    actions_per_state: int = 4,
    exploratory_sources: bool = False,
    exploratory_one_card_probabilities: dict[int, float] | None = None,
    learning_player_index: int = 0,
    player_count: int = 4,
    max_actions: int = 10_000,
    target_state: CounterfactualTargetState = "next-learner-turn",
) -> tuple[StateValueSample, ...]:
    """Collect value samples from sampled learner turn actions."""
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
    samples: list[StateValueSample] = []
    for source_index, state in enumerate(source_states):
        for action_index in _sample_action_indices(
            state,
            actions_per_state=actions_per_state,
            rng=rng,
        ):
            after_state = _apply_learner_turn_action(
                state,
                action_index=action_index,
                rng=rng,
            )
            sample_state = _counterfactual_sample_state(
                after_state,
                target_state=target_state,
                learning_player_index=learning_player_index,
                rng=rng,
                max_actions=max_actions,
            )
            final_state = _rollout_to_game_over(
                sample_state,
                rng=rng,
                max_actions=max_actions,
            )
            if final_state.phase != Phase.GAME_OVER:
                continue
            if not _can_evaluate_as_learner(
                sample_state,
                target_state=target_state,
                learning_player_index=learning_player_index,
            ):
                continue
            evaluated_state = _as_learner_perspective(
                sample_state,
                learning_player_index=learning_player_index,
            )
            player = evaluated_state.players[learning_player_index]
            samples.append(
                StateValueSample(
                    observation=normalize_observation(
                        state_to_observation(evaluated_state)
                    ),
                    target_loss_share=_loss_share(
                        final_state,
                        player_index=learning_player_index,
                    ),
                    player_index=learning_player_index,
                    hand_count=len(player.hand),
                    seed=source_seed_start + source_index,
                    turn_index=source_index,
                )
            )
    return tuple(samples)


def _counterfactual_sample_state(
    state: GameState,
    *,
    target_state: CounterfactualTargetState,
    learning_player_index: int,
    rng: Random,
    max_actions: int,
) -> GameState:
    if target_state == "after-refill":
        return state
    if target_state == "next-learner-turn":
        return _advance_to_learner_turn(
            state,
            learning_player_index=learning_player_index,
            rng=rng,
            max_actions=max_actions,
        )
    raise ValueError(f"unsupported counterfactual target state: {target_state}")


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
            player_index = state.current_player_index
            action = policies[player_index].choose_action(state)
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


def _apply_learner_turn_action(
    state: GameState,
    *,
    action_index: int,
    rng: Random,
) -> GameState:
    next_state = state
    for action in resolve_turn_action(next_state, action_index):
        next_state = apply_known_legal_action(next_state, action, rng=rng)
        if next_state.phase == Phase.GAME_OVER:
            break
    return next_state


def _advance_to_learner_turn(
    state: GameState,
    *,
    learning_player_index: int,
    rng: Random,
    max_actions: int,
) -> GameState:
    next_state = state
    action_count = 0
    bot = HeuristicBot()
    while (
        next_state.phase != Phase.GAME_OVER
        and not _is_learner_turn_start(next_state, learning_player_index)
        and action_count < max_actions
    ):
        action = bot.choose_action(next_state)
        if action is None:
            break
        next_state = apply_known_legal_action(next_state, action, rng=rng)
        action_count += 1
    return next_state


def _rollout_to_game_over(
    state: GameState,
    *,
    rng: Random,
    max_actions: int,
) -> GameState:
    next_state = state
    action_count = 0
    bot = HeuristicBot()
    while next_state.phase != Phase.GAME_OVER and action_count < max_actions:
        action = bot.choose_action(next_state)
        if action is None:
            break
        next_state = apply_known_legal_action(next_state, action, rng=rng)
        action_count += 1
    return next_state


def _is_learner_turn_start(state: GameState, learning_player_index: int) -> bool:
    return (
        state.phase == Phase.PLAY
        and state.current_player_index == learning_player_index
        and state.cards_played_this_turn == 0
    )


def _can_evaluate_as_learner(
    state: GameState,
    *,
    target_state: CounterfactualTargetState,
    learning_player_index: int,
) -> bool:
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        return False
    if target_state == "next-learner-turn":
        return state.current_player_index == learning_player_index
    return (
        state.current_player_index == learning_player_index
        or 0 <= learning_player_index < len(state.players)
    )


def _as_learner_perspective(
    state: GameState,
    *,
    learning_player_index: int,
) -> GameState:
    if state.current_player_index == learning_player_index:
        return state
    return replace(state, current_player_index=learning_player_index)


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


def _loss_share(state: GameState, *, player_index: int) -> float:
    total_loss = sum(player.loss_score for player in state.players)
    if total_loss == 0:
        return 0.0
    return state.players[player_index].loss_score / total_loss


def main() -> None:
    """CLI entry point for counterfactual state-value data collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-games", type=int, default=100)
    parser.add_argument("--source-seed-start", type=int, default=1)
    parser.add_argument("--source-state-limit", type=int, default=1000)
    parser.add_argument("--actions-per-state", type=int, default=4)
    parser.add_argument("--exploratory-sources", action="store_true")
    parser.add_argument(
        "--one-card-probability",
        action="append",
        metavar="HAND=PROBABILITY",
    )
    parser.add_argument(
        "--target-state",
        choices=("next-learner-turn", "after-refill"),
        default="next-learner-turn",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    samples = collect_counterfactual_state_value_samples(
        source_games=args.source_games,
        source_seed_start=args.source_seed_start,
        source_state_limit=args.source_state_limit,
        actions_per_state=args.actions_per_state,
        exploratory_sources=args.exploratory_sources,
        exploratory_one_card_probabilities=_parse_one_card_probabilities(
            tuple(args.one_card_probability or ())
        ),
        target_state=args.target_state,
    )
    write_state_value_samples(samples, args.output)
    summary = summarize_state_value_samples(
        samples,
        games=args.source_games,
        completed_games=len({sample.seed for sample in samples}),
    )
    summary_dict = state_value_dataset_summary_to_dict(summary)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
