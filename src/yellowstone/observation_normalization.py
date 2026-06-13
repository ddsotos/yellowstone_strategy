"""Observation bounds and normalization helpers for learning libraries."""

from __future__ import annotations

from yellowstone.observation import (
    BOARD_OBSERVATION_SIZE,
    COLOR_ORDER,
    CURRENT_PLAYER_OBSERVATION_SIZE,
    HAND_OBSERVATION_SIZE,
    HAND_SLOT_FEATURE_SIZE,
    OBSERVATION_SIZE,
    PHASE_OBSERVATION_SIZE,
    PLAYER_FEATURE_SIZE,
    PLAYERS_OBSERVATION_SIZE,
    SCALAR_OBSERVATION_SIZE,
)
from yellowstone.types import BOARD_SIZE, DEFAULT_LOSS_SCORE, HAND_SIZE, MAX_PLAYERS


MAX_CARD_COPIES_PER_COLOR = BOARD_SIZE * 2
MAX_DECK_SIZE = BOARD_SIZE * len(COLOR_ORDER) * 2
MAX_OBSERVED_LOSS_SCORE = 64
MAX_SETTLEMENT_COUNT = 64


def observation_high_values() -> tuple[int, ...]:
    """Return per-feature upper bounds for raw integer observations."""
    values: list[int] = []
    values.extend([MAX_CARD_COPIES_PER_COLOR] * BOARD_OBSERVATION_SIZE)
    for _ in range(HAND_SIZE):
        values.append(1)
        values.extend([1] * len(COLOR_ORDER))
        values.append(BOARD_SIZE - 1)
    for _ in range(MAX_PLAYERS):
        values.extend(
            [
                1,
                HAND_SIZE,
                MAX_DECK_SIZE,
                MAX_OBSERVED_LOSS_SCORE,
            ]
        )
    values.extend([1] * CURRENT_PLAYER_OBSERVATION_SIZE)
    values.extend([1] * PHASE_OBSERVATION_SIZE)
    values.extend(
        [
            2,
            MAX_DECK_SIZE,
            MAX_SETTLEMENT_COUNT,
            MAX_PLAYERS,
        ]
    )
    if len(values) != OBSERVATION_SIZE:
        raise AssertionError(f"unexpected observation bounds size: {len(values)}")
    return tuple(values)


def normalize_observation(observation: tuple[int, ...]) -> tuple[float, ...]:
    """Scale a raw observation to roughly 0..1 per feature."""
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_SIZE}")
    highs = observation_high_values()
    return tuple(
        0.0 if high == 0 else min(float(value) / float(high), 1.0)
        for value, high in zip(observation, highs, strict=True)
    )


def observation_normalization_policy() -> dict[str, int | str]:
    """Return the documented normalization policy for training code."""
    return {
        "core_api": "raw_integer_tuple",
        "gym_api": "float32_normalized_by_default",
        "observation_size": OBSERVATION_SIZE,
        "board_stack_color_max": MAX_CARD_COPIES_PER_COLOR,
        "deck_size_max": MAX_DECK_SIZE,
        "loss_score_clip_max": MAX_OBSERVED_LOSS_SCORE,
        "initial_loss_score": DEFAULT_LOSS_SCORE,
        "hand_slot_feature_size": HAND_SLOT_FEATURE_SIZE,
        "hand_observation_size": HAND_OBSERVATION_SIZE,
        "player_feature_size": PLAYER_FEATURE_SIZE,
        "players_observation_size": PLAYERS_OBSERVATION_SIZE,
        "scalar_observation_size": SCALAR_OBSERVATION_SIZE,
    }
