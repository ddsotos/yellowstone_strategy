"""Fixed-length numeric observation encoding for reinforcement learning."""

from __future__ import annotations

from yellowstone.types import (
    BOARD_SIZE,
    HAND_SIZE,
    MAX_PLAYERS,
    Card,
    Color,
    GameState,
    Phase,
    Position,
)


COLOR_ORDER = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)
PHASE_ORDER = (Phase.PLAY, Phase.REFILL, Phase.GAME_OVER)
CELL_FEATURE_SIZE = len(COLOR_ORDER)
HAND_SLOT_FEATURE_SIZE = 1 + len(COLOR_ORDER) + 1
PLAYER_FEATURE_SIZE = 4
BOARD_OBSERVATION_SIZE = BOARD_SIZE * BOARD_SIZE * CELL_FEATURE_SIZE
HAND_OBSERVATION_SIZE = HAND_SIZE * HAND_SLOT_FEATURE_SIZE
PLAYERS_OBSERVATION_SIZE = MAX_PLAYERS * PLAYER_FEATURE_SIZE
CURRENT_PLAYER_OBSERVATION_SIZE = MAX_PLAYERS
PHASE_OBSERVATION_SIZE = len(PHASE_ORDER)
SCALAR_OBSERVATION_SIZE = 4
OBSERVATION_SIZE = (
    BOARD_OBSERVATION_SIZE
    + HAND_OBSERVATION_SIZE
    + PLAYERS_OBSERVATION_SIZE
    + CURRENT_PLAYER_OBSERVATION_SIZE
    + PHASE_OBSERVATION_SIZE
    + SCALAR_OBSERVATION_SIZE
)


def state_to_observation(state: GameState) -> tuple[int, ...]:
    """Encode a state as a fixed-length integer tuple."""
    values: list[int] = []
    values.extend(_board_features(state))
    values.extend(_current_hand_features(state))
    values.extend(_player_features(state))
    values.extend(_one_hot(state.current_player_index, MAX_PLAYERS))
    values.extend(_one_hot(PHASE_ORDER.index(state.phase), len(PHASE_ORDER)))
    values.extend(
        [
            state.cards_played_this_turn,
            len(state.deck),
            state.settlement_count,
            len(state.players),
        ]
    )
    if len(values) != OBSERVATION_SIZE:
        raise AssertionError(f"unexpected observation size: {len(values)}")
    return tuple(values)


def observation_metadata() -> dict[str, int]:
    """Return feature sizes for downstream encoders and tests."""
    return {
        "observation_size": OBSERVATION_SIZE,
        "board_size": BOARD_OBSERVATION_SIZE,
        "hand_size": HAND_OBSERVATION_SIZE,
        "players_size": PLAYERS_OBSERVATION_SIZE,
        "current_player_size": CURRENT_PLAYER_OBSERVATION_SIZE,
        "phase_size": PHASE_OBSERVATION_SIZE,
        "scalar_size": SCALAR_OBSERVATION_SIZE,
    }


def _board_features(state: GameState) -> list[int]:
    values: list[int] = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            stack = state.board.get(Position(x=x, y=y), ())
            values.extend(_stack_color_counts(stack))
    return values


def _current_hand_features(state: GameState) -> list[int]:
    values: list[int] = []
    player = state.players[state.current_player_index]
    for index in range(HAND_SIZE):
        if index < len(player.hand):
            card = player.hand[index]
            values.append(1)
            values.extend(_one_hot(COLOR_ORDER.index(card.color), len(COLOR_ORDER)))
            values.append(card.rank_index)
        else:
            values.extend([0] * HAND_SLOT_FEATURE_SIZE)
    return values


def _player_features(state: GameState) -> list[int]:
    values: list[int] = []
    for index in range(MAX_PLAYERS):
        if index < len(state.players):
            player = state.players[index]
            values.extend(
                [
                    1,
                    len(player.hand),
                    len(player.negative_cards),
                    player.loss_score,
                ]
            )
        else:
            values.extend([0] * PLAYER_FEATURE_SIZE)
    return values


def _stack_color_counts(stack: tuple[Card, ...]) -> list[int]:
    return [sum(1 for card in stack if card.color == color) for color in COLOR_ORDER]


def _one_hot(index: int, size: int) -> list[int]:
    return [1 if current == index else 0 for current in range(size)]
