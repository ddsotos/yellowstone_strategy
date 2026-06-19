"""Collect action-value samples for turn-level learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Iterable

from yellowstone.bots import BotPolicy, ExploratoryHeuristicBot, HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.policy_diagnostics import _heuristic_turn_action_index
from yellowstone.turn_action_space import (
    TURN_ACTION_SPACE_SIZE,
    legal_turn_action_indices,
    resolve_turn_action_before_refill,
)
from yellowstone.types import GameState, Phase
from yellowstone.types import HAND_SIZE, RefillAction, RefillSource


@dataclass(frozen=True, slots=True)
class ActionValueSample:
    """One turn-start action labeled by finite-horizon future loss."""

    observation: tuple[float, ...]
    after_observation: tuple[float, ...]
    action_index: int
    target_self_loss: float
    target_relative_loss: float
    player_index: int
    hand_count: int
    seed: int
    turn_index: int
    heuristic_action_index: int = -1
    target_self_advantage: float = 0.0
    target_relative_advantage: float = 0.0


@dataclass(frozen=True, slots=True)
class ActionValueDatasetSummary:
    """Summary of generated action-value samples."""

    source_games: int
    source_states: int
    samples: int
    horizon_learner_turns: int
    hand_count_histogram: tuple[int, ...]
    action_count_histogram: tuple[int, ...]
    continuing_game: bool = False


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
    continuing_game: bool = False,
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
        continuing_game=continuing_game,
    )
    samples: list[ActionValueSample] = []
    for source_index, state in enumerate(source_states):
        player = state.players[learning_player_index]
        observation = normalize_observation(state_to_observation(state))
        baseline_losses = tuple(player.loss_score for player in state.players)
        heuristic_action_index = _heuristic_turn_action_index(state)
        action_results: list[tuple[int, GameState, tuple[float, float]]] = []
        for action_index in _sample_action_indices(
            state,
            actions_per_state=actions_per_state,
            rng=rng,
            required_action_index=heuristic_action_index,
        ):
            after_decision = _apply_turn_action_before_refill(
                state,
                action_index=action_index,
                rng=rng,
            )
            result = _rollout_horizon(
                after_decision,
                baseline_losses=baseline_losses,
                learning_player_index=learning_player_index,
                horizon_learner_turns=horizon_learner_turns,
                rng=rng,
                max_actions=max_actions,
                continuing_game=continuing_game,
            )
            if result is None:
                continue
            action_results.append((action_index, after_decision, result))
        heuristic_result = next(
            (
                result
                for action_index, _, result in action_results
                if action_index == heuristic_action_index
            ),
            None,
        )
        if heuristic_result is None:
            continue
        heuristic_self_loss, heuristic_relative_loss = heuristic_result
        for action_index, after_decision, result in action_results:
            target_self_loss, target_relative_loss = result
            samples.append(
                ActionValueSample(
                    observation=observation,
                    after_observation=_after_decision_observation(
                        after_decision,
                        learning_player_index=learning_player_index,
                    ),
                    action_index=action_index,
                    target_self_loss=target_self_loss,
                    target_relative_loss=target_relative_loss,
                    player_index=learning_player_index,
                    hand_count=len(player.hand),
                    seed=source_seed_start + source_index,
                    turn_index=source_index,
                    heuristic_action_index=heuristic_action_index,
                    target_self_advantage=heuristic_self_loss - target_self_loss,
                    target_relative_advantage=(
                        heuristic_relative_loss - target_relative_loss
                    ),
                )
            )
    return tuple(samples)


def summarize_action_value_samples(
    samples: Iterable[ActionValueSample],
    *,
    source_games: int,
    source_states: int,
    horizon_learner_turns: int,
    continuing_game: bool = False,
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
        continuing_game=continuing_game,
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
        "continuing_game": summary.continuing_game,
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
    continuing_game: bool,
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
        while (continuing_game or state.phase != Phase.GAME_OVER) and action_count < max_actions:
            if _is_learner_turn_start(state, learning_player_index):
                source_states.append(state)
                if len(source_states) >= source_state_limit:
                    break
            action = policies[state.current_player_index].choose_action(state)
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
    required_action_index: int | None = None,
) -> tuple[int, ...]:
    legal_indices = list(legal_turn_action_indices(state))
    if actions_per_state <= 0 or actions_per_state >= len(legal_indices):
        return tuple(legal_indices)
    rng.shuffle(legal_indices)
    selected = set(legal_indices[:actions_per_state])
    if required_action_index is not None and required_action_index in legal_indices:
        selected.add(required_action_index)
    return tuple(sorted(selected))


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


def _after_decision_observation(
    state: GameState,
    *,
    learning_player_index: int,
) -> tuple[float, ...]:
    learner_perspective = replace(state, current_player_index=learning_player_index)
    return normalize_observation(state_to_observation(learner_perspective))


def _rollout_horizon(
    state: GameState,
    *,
    baseline_losses: tuple[int, ...],
    learning_player_index: int,
    horizon_learner_turns: int,
    rng: Random,
    max_actions: int,
    continuing_game: bool,
) -> tuple[float, float] | None:
    next_state = state
    bot = HeuristicBot()
    action_count = 0
    learner_turns = 0
    while (
        (continuing_game or next_state.phase != Phase.GAME_OVER)
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
        next_state = _apply_rollout_action(
            next_state,
            action,
            rng=rng,
            continuing_game=continuing_game,
        )
        action_count += 1
    if learner_turns < horizon_learner_turns:
        return None
    loss_deltas = tuple(
        player.loss_score - baseline_loss
        for player, baseline_loss in zip(
            next_state.players,
            baseline_losses,
            strict=True,
        )
    )
    self_loss = float(loss_deltas[learning_player_index])
    average_loss = sum(loss_deltas) / len(loss_deltas)
    return self_loss, self_loss - average_loss


def _apply_rollout_action(
    state: GameState,
    action,
    *,
    rng: Random,
    continuing_game: bool,
) -> GameState:
    if not continuing_game:
        return apply_known_legal_action(state, action, rng=rng)
    if isinstance(action, RefillAction) and action.source == RefillSource.DECK:
        player = state.players[state.current_player_index]
        if len(state.deck) < max(0, HAND_SIZE - len(player.hand)):
            return _apply_continuing_deck_refill(state, rng=rng)
    next_state = apply_known_legal_action(state, action, rng=rng)
    if next_state.phase == Phase.GAME_OVER:
        return _continue_after_game_over(next_state, rng=rng)
    return next_state


def _apply_continuing_deck_refill(state: GameState, *, rng: Random) -> GameState:
    player = state.players[state.current_player_index]
    deck = list(state.deck)
    hand = list(player.hand)
    draw_count = max(0, HAND_SIZE - len(hand))
    for _ in range(min(draw_count, len(deck))):
        hand.append(deck.pop(0))
    players = list(state.players)
    players[state.current_player_index] = replace(player, hand=tuple(hand))
    return _continue_after_deck_exhaustion(
        replace(state, players=tuple(players), deck=tuple(deck)),
        rng=rng,
    )


def _continue_after_deck_exhaustion(state: GameState, *, rng: Random) -> GameState:
    settled_players = []
    collected_negative_cards = []
    for player in state.players:
        collected_negative_cards.extend(player.negative_cards)
        settled_players.append(
            replace(
                player,
                negative_cards=(),
                loss_score=player.loss_score + len(player.negative_cards),
            )
        )
    rng.shuffle(collected_negative_cards)
    return replace(
        state,
        players=tuple(settled_players),
        deck=tuple(collected_negative_cards),
        current_player_index=(state.current_player_index + 1) % len(state.players),
        phase=Phase.PLAY,
        cards_played_this_turn=0,
        winners=(),
        settlement_count=state.settlement_count + 1,
    )


def _continue_after_game_over(state: GameState, *, rng: Random) -> GameState:
    negative_cards = [
        card
        for player in state.players
        for card in player.negative_cards
    ]
    if negative_cards and not state.deck:
        rng.shuffle(negative_cards)
        players = tuple(replace(player, negative_cards=()) for player in state.players)
        return replace(
            state,
            players=players,
            deck=tuple(negative_cards),
            phase=Phase.PLAY,
            winners=(),
            cards_played_this_turn=0,
        )
    return replace(
        state,
        phase=Phase.PLAY,
        winners=(),
        cards_played_this_turn=0,
    )


def _is_learner_turn_start(state: GameState, learning_player_index: int) -> bool:
    return (
        state.phase == Phase.PLAY
        and state.current_player_index == learning_player_index
        and state.cards_played_this_turn == 0
    )


def _sample_to_dict(sample: ActionValueSample) -> dict[str, object]:
    return {
        "observation": list(sample.observation),
        "after_observation": list(sample.after_observation),
        "action_index": sample.action_index,
        "target_self_loss": sample.target_self_loss,
        "target_relative_loss": sample.target_relative_loss,
        "player_index": sample.player_index,
        "hand_count": sample.hand_count,
        "seed": sample.seed,
        "turn_index": sample.turn_index,
        "heuristic_action_index": sample.heuristic_action_index,
        "target_self_advantage": sample.target_self_advantage,
        "target_relative_advantage": sample.target_relative_advantage,
    }


def _sample_from_dict(data: dict[str, object]) -> ActionValueSample:
    observation = tuple(float(value) for value in data["observation"])  # type: ignore[index]
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_SIZE}")
    raw_after_observation = data.get("after_observation", data["observation"])
    after_observation = tuple(float(value) for value in raw_after_observation)  # type: ignore[arg-type]
    if len(after_observation) != OBSERVATION_SIZE:
        raise ValueError(f"after_observation must have length {OBSERVATION_SIZE}")
    action_index = int(data["action_index"])
    if not 0 <= action_index < TURN_ACTION_SPACE_SIZE:
        raise ValueError(f"action index out of range: {action_index}")
    return ActionValueSample(
        observation=observation,
        after_observation=after_observation,
        action_index=action_index,
        target_self_loss=float(data["target_self_loss"]),
        target_relative_loss=float(data["target_relative_loss"]),
        player_index=int(data["player_index"]),
        hand_count=int(data["hand_count"]),
        seed=int(data["seed"]),
        turn_index=int(data["turn_index"]),
        heuristic_action_index=int(data.get("heuristic_action_index", -1)),
        target_self_advantage=float(data.get("target_self_advantage", 0.0)),
        target_relative_advantage=float(
            data.get("target_relative_advantage", 0.0)
        ),
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
    parser.add_argument("--continuing-game", action="store_true")
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
        continuing_game=args.continuing_game,
    )
    write_action_value_samples(samples, args.output)
    summary = summarize_action_value_samples(
        samples,
        source_games=args.source_games,
        source_states=len({sample.turn_index for sample in samples}),
        horizon_learner_turns=args.horizon_learner_turns,
        continuing_game=args.continuing_game,
    )
    summary_dict = action_value_dataset_summary_to_dict(summary)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
