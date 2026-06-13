"""Batch match evaluation for bot policies."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Iterable

from yellowstone.bots import BotPolicy, HeuristicBot, RandomBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.types import GameState, Phase


@dataclass(frozen=True, slots=True)
class MatchResult:
    seed: int
    winners: tuple[int, ...]
    loss_scores: tuple[int, ...]
    turn_count: int
    action_count: int
    elapsed_seconds: float
    game_over: bool


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    match_count: int
    win_rates: tuple[float, ...]
    average_loss_scores: tuple[float, ...]
    average_turn_count: float
    total_elapsed_seconds: float
    average_elapsed_seconds: float


def run_match(
    policies: tuple[BotPolicy, ...],
    *,
    seed: int,
    max_actions: int = 10_000,
) -> MatchResult:
    """Run one complete match for the given policies."""
    state = create_initial_state(len(policies), seed=seed)
    rng = Random(seed)
    action_count = 0
    turn_count = 0
    started_at = perf_counter()

    while state.phase != Phase.GAME_OVER and action_count < max_actions:
        player_index = state.current_player_index
        action = policies[player_index].choose_action(state)
        if action is None:
            break
        state = apply_known_legal_action(state, action, rng=rng)
        action_count += 1
        if state.current_player_index != player_index:
            turn_count += 1

    elapsed_seconds = perf_counter() - started_at
    return _result_from_state(
        state,
        seed=seed,
        turn_count=turn_count,
        action_count=action_count,
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_policies(
    policies: tuple[BotPolicy, ...],
    *,
    seeds: Iterable[int],
    max_actions: int = 10_000,
) -> tuple[EvaluationSummary, tuple[MatchResult, ...]]:
    """Run policies across seeds and aggregate evaluation metrics."""
    results = tuple(
        run_match(policies, seed=seed, max_actions=max_actions) for seed in seeds
    )
    return summarize_results(results, player_count=len(policies)), results


def summarize_results(
    results: tuple[MatchResult, ...],
    *,
    player_count: int,
) -> EvaluationSummary:
    """Aggregate match results into win rates and averages."""
    if not results:
        return EvaluationSummary(
            match_count=0,
            win_rates=tuple(0.0 for _ in range(player_count)),
            average_loss_scores=tuple(0.0 for _ in range(player_count)),
            average_turn_count=0.0,
            total_elapsed_seconds=0.0,
            average_elapsed_seconds=0.0,
        )

    match_count = len(results)
    win_rates = tuple(
        sum(1 for result in results if player_index in result.winners) / match_count
        for player_index in range(player_count)
    )
    average_loss_scores = tuple(
        sum(result.loss_scores[player_index] for result in results) / match_count
        for player_index in range(player_count)
    )
    total_elapsed_seconds = sum(result.elapsed_seconds for result in results)
    return EvaluationSummary(
        match_count=match_count,
        win_rates=win_rates,
        average_loss_scores=average_loss_scores,
        average_turn_count=sum(result.turn_count for result in results) / match_count,
        total_elapsed_seconds=total_elapsed_seconds,
        average_elapsed_seconds=total_elapsed_seconds / match_count,
    )


def make_heuristic_policies(player_count: int = 4) -> tuple[HeuristicBot, ...]:
    """Create deterministic heuristic policies for all players."""
    return tuple(HeuristicBot() for _ in range(player_count))


def make_random_policies(
    *,
    player_count: int = 4,
    seed: int = 0,
) -> tuple[RandomBot, ...]:
    """Create seeded random policies for all players."""
    return tuple(RandomBot(Random(seed + index)) for index in range(player_count))


def _result_from_state(
    state: GameState,
    *,
    seed: int,
    turn_count: int,
    action_count: int,
    elapsed_seconds: float,
) -> MatchResult:
    return MatchResult(
        seed=seed,
        winners=state.winners,
        loss_scores=tuple(player.loss_score for player in state.players),
        turn_count=turn_count,
        action_count=action_count,
        elapsed_seconds=elapsed_seconds,
        game_over=state.phase == Phase.GAME_OVER,
    )
