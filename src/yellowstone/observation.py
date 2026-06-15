"""Fixed-length numeric observation encoding for reinforcement learning."""

from __future__ import annotations

from yellowstone.types import (
    FRAME_SIZE,
    HAND_SIZE,
    Card,
    Color,
    GameState,
    Position,
)


COLOR_ORDER = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)
RELATIVE_COLOR_FEATURE_SIZE = len(COLOR_ORDER)
HAND_SLOT_FEATURE_SIZE = 1 + RELATIVE_COLOR_FEATURE_SIZE + 1
OBSERVED_PLAYER_COUNT = 4
PLAYER_FEATURE_SIZE = 3
BOARD_ANCHOR_FEATURE_SIZE = 2
BOARD_CELL_COUNT_FEATURE_SIZE = FRAME_SIZE * FRAME_SIZE
BOARD_OBSERVATION_SIZE = BOARD_ANCHOR_FEATURE_SIZE + BOARD_CELL_COUNT_FEATURE_SIZE
HAND_OBSERVATION_SIZE = HAND_SIZE * HAND_SLOT_FEATURE_SIZE
PLAYERS_OBSERVATION_SIZE = OBSERVED_PLAYER_COUNT * PLAYER_FEATURE_SIZE
SCALAR_OBSERVATION_SIZE = 2
OBSERVATION_SIZE = (
    BOARD_OBSERVATION_SIZE
    + HAND_OBSERVATION_SIZE
    + PLAYERS_OBSERVATION_SIZE
    + SCALAR_OBSERVATION_SIZE
)


def state_to_observation(state: GameState) -> tuple[int, ...]:
    """Encode a state as a fixed-length integer tuple."""
    values: list[int] = []
    values.extend(_board_features(state))
    values.extend(_current_hand_features(state))
    values.extend(_player_features(state))
    values.extend(
        [
            _deck_bucket(len(state.deck)),
            state.settlement_count,
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
        "scalar_size": SCALAR_OBSERVATION_SIZE,
    }


def _board_features(state: GameState) -> list[int]:
    anchor = _board_anchor(state)
    values: list[int] = [anchor.y, anchor.x]
    for y in range(anchor.y, anchor.y + FRAME_SIZE):
        for x in range(anchor.x, anchor.x + FRAME_SIZE):
            values.append(len(state.board.get(Position(x=x, y=y), ())))
    return values


def _board_anchor(state: GameState) -> Position:
    if not state.board:
        return Position(x=0, y=0)
    min_x = min(position.x for position in state.board)
    min_y = min(position.y for position in state.board)
    return Position(x=min_x, y=min_y)


def _current_hand_features(state: GameState) -> list[int]:
    values: list[int] = []
    player = state.players[state.current_player_index]
    relative_color_indices = _relative_color_indices(state)
    for index in range(HAND_SIZE):
        if index < len(player.hand):
            card = player.hand[index]
            values.append(1)
            values.extend(
                _one_hot(
                    relative_color_indices[card.color],
                    RELATIVE_COLOR_FEATURE_SIZE,
                )
            )
            values.append(card.rank_index)
        else:
            values.extend([0] * HAND_SLOT_FEATURE_SIZE)
    return values


def _player_features(state: GameState) -> list[int]:
    values: list[int] = []
    for index in range(OBSERVED_PLAYER_COUNT):
        if index < len(state.players):
            player = state.players[index]
            values.extend(
                [
                    len(player.hand),
                    len(player.negative_cards),
                    player.loss_score,
                ]
            )
        else:
            values.extend([0] * PLAYER_FEATURE_SIZE)
    return values


def _deck_bucket(deck_count: int) -> int:
    if deck_count == 0:
        return 0
    if deck_count <= HAND_SIZE:
        return 1
    if deck_count <= HAND_SIZE * 3:
        return 2
    return 3


def _relative_color_indices(state: GameState) -> dict[Color, int]:
    anchor = _board_anchor(state)
    assigned: dict[Color, int] = {}
    used_indices: set[int] = set()

    for relative_index, x in enumerate(
        range(anchor.x + FRAME_SIZE - 1, anchor.x - 1, -1)
    ):
        for color in _colors_in_column(state, x, anchor.y):
            assigned[color] = relative_index
            used_indices.add(relative_index)

    player = state.players[state.current_player_index]
    for card in player.hand:
        if card.color in assigned:
            continue
        relative_index = _first_unused_relative_color_index(used_indices)
        assigned[card.color] = relative_index
        used_indices.add(relative_index)

    for color in COLOR_ORDER:
        if color in assigned:
            continue
        relative_index = _first_unused_relative_color_index(used_indices)
        assigned[color] = relative_index
        used_indices.add(relative_index)
    return assigned


def _colors_in_column(state: GameState, x: int, min_y: int) -> tuple[Color, ...]:
    colors = {
        card.color
        for y in range(min_y, min_y + FRAME_SIZE)
        for card in state.board.get(Position(x=x, y=y), ())
    }
    return tuple(color for color in COLOR_ORDER if color in colors)


def _first_unused_relative_color_index(used_indices: set[int]) -> int:
    for index in range(RELATIVE_COLOR_FEATURE_SIZE):
        if index not in used_indices:
            return index
    raise AssertionError("relative color index exhausted")


def _one_hot(index: int, size: int) -> list[int]:
    return [1 if current == index else 0 for current in range(size)]
