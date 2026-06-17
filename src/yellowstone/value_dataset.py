"""Collect turn-start state-value samples from bot rollouts."""

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
from yellowstone.types import GameState, Phase


@dataclass(frozen=True, slots=True)
class StateValueSample:
    """One turn-start observation labeled by final loss share."""

    observation: tuple[float, ...]
    target_loss_share: float
    player_index: int
    hand_count: int
    seed: int
    turn_index: int


@dataclass(frozen=True, slots=True)
class StateValueDatasetSummary:
    """Summary of generated state-value samples."""

    games: int
    completed_games: int
    samples: int
    hand_count_histogram: tuple[int, ...]


def collect_state_value_samples(
    *,
    games: int,
    seed_start: int = 1,
    player_count: int = 4,
    exploratory: bool = False,
    max_actions: int = 10_000,
) -> tuple[StateValueSample, ...]:
    """Collect turn-start observations and label them with final loss share."""
    samples: list[StateValueSample] = []
    for offset in range(games):
        seed = seed_start + offset
        policies = _make_policies(
            player_count=player_count,
            exploratory=exploratory,
            seed=seed,
        )
        samples.extend(
            _collect_game_samples(
                policies,
                seed=seed,
                max_actions=max_actions,
            )
        )
    return tuple(samples)


def summarize_state_value_samples(
    samples: Iterable[StateValueSample],
    *,
    games: int,
    completed_games: int,
) -> StateValueDatasetSummary:
    """Summarize sample counts and hand-count coverage."""
    sample_tuple = tuple(samples)
    histogram = [0] * 7
    for sample in sample_tuple:
        if 0 <= sample.hand_count < len(histogram):
            histogram[sample.hand_count] += 1
    return StateValueDatasetSummary(
        games=games,
        completed_games=completed_games,
        samples=len(sample_tuple),
        hand_count_histogram=tuple(histogram),
    )


def write_state_value_samples(
    samples: Iterable[StateValueSample],
    path: str | Path,
) -> None:
    """Write samples as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(_sample_to_dict(sample), separators=(",", ":")))
            file.write("\n")


def read_state_value_samples(path: str | Path) -> tuple[StateValueSample, ...]:
    """Read samples from JSON Lines."""
    samples: list[StateValueSample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            samples.append(_sample_from_dict(json.loads(line)))
    return tuple(samples)


def state_value_dataset_summary_to_dict(
    summary: StateValueDatasetSummary,
) -> dict[str, int | list[int]]:
    """Convert a dataset summary to JSON-friendly data."""
    return {
        "games": summary.games,
        "completed_games": summary.completed_games,
        "samples": summary.samples,
        "hand_count_histogram": list(summary.hand_count_histogram),
    }


def _collect_game_samples(
    policies: tuple[BotPolicy, ...],
    *,
    seed: int,
    max_actions: int,
) -> tuple[StateValueSample, ...]:
    state = create_initial_state(len(policies), seed=seed)
    rng = Random(seed)
    pending: list[tuple[tuple[float, ...], int, int, int]] = []
    action_count = 0
    turn_index = 0

    while state.phase != Phase.GAME_OVER and action_count < max_actions:
        if _is_turn_start(state):
            player_index = state.current_player_index
            player = state.players[player_index]
            pending.append(
                (
                    normalize_observation(state_to_observation(state)),
                    player_index,
                    len(player.hand),
                    turn_index,
                )
            )
            turn_index += 1

        player_index = state.current_player_index
        action = policies[player_index].choose_action(state)
        if action is None:
            break
        state = apply_known_legal_action(state, action, rng=rng)
        action_count += 1

    if state.phase != Phase.GAME_OVER:
        return ()

    loss_shares = _loss_shares(state)
    return tuple(
        StateValueSample(
            observation=observation,
            target_loss_share=loss_shares[player_index],
            player_index=player_index,
            hand_count=hand_count,
            seed=seed,
            turn_index=sample_turn_index,
        )
        for observation, player_index, hand_count, sample_turn_index in pending
    )


def _make_policies(
    *,
    player_count: int,
    exploratory: bool,
    seed: int,
) -> tuple[BotPolicy, ...]:
    if exploratory:
        return tuple(
            ExploratoryHeuristicBot(rng=Random(seed + player_index))
            for player_index in range(player_count)
        )
    return tuple(HeuristicBot() for _ in range(player_count))


def _is_turn_start(state: GameState) -> bool:
    return state.phase == Phase.PLAY and state.cards_played_this_turn == 0


def _loss_shares(state: GameState) -> tuple[float, ...]:
    total_loss = sum(player.loss_score for player in state.players)
    if total_loss == 0:
        return tuple(0.0 for _ in state.players)
    return tuple(player.loss_score / total_loss for player in state.players)


def _sample_to_dict(sample: StateValueSample) -> dict[str, object]:
    return {
        "observation": list(sample.observation),
        "target_loss_share": sample.target_loss_share,
        "player_index": sample.player_index,
        "hand_count": sample.hand_count,
        "seed": sample.seed,
        "turn_index": sample.turn_index,
    }


def _sample_from_dict(data: dict[str, object]) -> StateValueSample:
    observation = tuple(float(value) for value in data["observation"])  # type: ignore[index]
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_SIZE}")
    return StateValueSample(
        observation=observation,
        target_loss_share=float(data["target_loss_share"]),
        player_index=int(data["player_index"]),
        hand_count=int(data["hand_count"]),
        seed=int(data["seed"]),
        turn_index=int(data["turn_index"]),
    )


def main() -> None:
    """CLI entry point for state-value dataset collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--max-actions", type=int, default=10_000)
    args = parser.parse_args()

    samples = collect_state_value_samples(
        games=args.games,
        seed_start=args.seed_start,
        exploratory=args.exploratory,
        max_actions=args.max_actions,
    )
    write_state_value_samples(samples, args.output)
    summary = summarize_state_value_samples(
        samples,
        games=args.games,
        completed_games=len({sample.seed for sample in samples}),
    )
    summary_dict = state_value_dataset_summary_to_dict(summary)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary_dict, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
