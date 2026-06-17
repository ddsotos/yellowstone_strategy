"""Turn-level action mapping for simpler reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass

from yellowstone.bots import HeuristicBot, placement_sort_key
from yellowstone.game import apply_known_legal_action, legal_actions
from yellowstone.types import (
    HAND_SIZE,
    Action,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
)


TURN_ACTION_SPACE_SIZE = HAND_SIZE + HAND_SIZE * (HAND_SIZE - 1)


@dataclass(frozen=True, slots=True)
class TurnAction:
    """A turn-level choice of one or two original hand slots."""

    hand_indices: tuple[int, ...]


def turn_action_from_index(index: int) -> TurnAction:
    """Decode a fixed turn-level action index."""
    if not 0 <= index < TURN_ACTION_SPACE_SIZE:
        raise ValueError(f"turn action index out of range: {index}")
    if index < HAND_SIZE:
        return TurnAction((index,))
    first_index, second_offset = divmod(index - HAND_SIZE, HAND_SIZE - 1)
    second_index = (
        second_offset
        if second_offset < first_index
        else second_offset + 1
    )
    return TurnAction((first_index, second_index))


def turn_action_to_index(action: TurnAction) -> int:
    """Encode a one-card or two-card turn choice."""
    if len(action.hand_indices) == 1:
        hand_index = action.hand_indices[0]
        if not 0 <= hand_index < HAND_SIZE:
            raise ValueError(f"hand index out of range: {hand_index}")
        return hand_index
    if len(action.hand_indices) == 2:
        first_index, second_index = action.hand_indices
        if first_index == second_index:
            raise ValueError("two-card turn action must use two different slots")
        if not 0 <= first_index < HAND_SIZE:
            raise ValueError(f"hand index out of range: {first_index}")
        if not 0 <= second_index < HAND_SIZE:
            raise ValueError(f"hand index out of range: {second_index}")
        second_offset = second_index if second_index < first_index else second_index - 1
        return HAND_SIZE + first_index * (HAND_SIZE - 1) + second_offset
    raise ValueError("turn action must contain one or two hand indices")


def resolve_turn_action(state: GameState, index: int) -> tuple[Action, ...]:
    """Resolve a turn-level action into concrete heuristic low-level actions."""
    resolved = list(resolve_turn_action_before_refill(state, index))
    current_state = state
    for action in resolved:
        current_state = apply_known_legal_action(current_state, action)

    if current_state.phase == Phase.REFILL:
        refill_action = HeuristicBot().choose_action(current_state)
        if refill_action is None:
            raise ValueError("no refill action is available")
        resolved.append(refill_action)
    return tuple(resolved)


def resolve_turn_action_before_refill(
    state: GameState,
    index: int,
) -> tuple[Action, ...]:
    """Resolve a turn-level action, stopping before refill randomness."""
    action = turn_action_from_index(index)
    if state.phase != Phase.PLAY:
        raise ValueError("turn action can only start in play phase")
    if state.cards_played_this_turn != 0:
        raise ValueError("turn action can only start at the beginning of a turn")

    current_state = state
    resolved: list[Action] = []
    placed_original_indices: list[int] = []
    for original_index in action.hand_indices:
        hand_index = _current_hand_index(original_index, placed_original_indices)
        place_action = _choose_place_action_for_hand_index(current_state, hand_index)
        resolved.append(place_action)
        placed_original_indices.append(original_index)
        current_state = apply_known_legal_action(current_state, place_action)

    if len(action.hand_indices) == 1:
        end_turn = EndTurnAction()
        if end_turn not in legal_actions(current_state):
            raise ValueError("end turn is not legal after selected card")
        resolved.append(end_turn)
        current_state = apply_known_legal_action(current_state, end_turn)

    return tuple(resolved)


def legal_turn_action_indices(state: GameState) -> tuple[int, ...]:
    """Return legal turn-level action indexes."""
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        return ()

    hand_count = len(state.players[state.current_player_index].hand)
    indexes: list[int] = []
    indexes.extend(range(min(hand_count, HAND_SIZE)))
    for first_index in range(min(hand_count, HAND_SIZE)):
        for second_index in range(min(hand_count, HAND_SIZE)):
            if first_index == second_index:
                continue
            indexes.append(turn_action_to_index(TurnAction((first_index, second_index))))
    return tuple(indexes)


def legal_turn_action_mask(state: GameState) -> tuple[bool, ...]:
    """Return a fixed-size legal mask for turn-level actions."""
    legal_indexes = set(legal_turn_action_indices(state))
    return tuple(index in legal_indexes for index in range(TURN_ACTION_SPACE_SIZE))


def turn_action_space_metadata() -> dict[str, int]:
    """Return turn-level action space metadata."""
    return {
        "turn_action_space_size": TURN_ACTION_SPACE_SIZE,
        "one_card_action_count": HAND_SIZE,
        "two_card_action_count": HAND_SIZE * (HAND_SIZE - 1),
    }


def _choose_place_action_for_hand_index(
    state: GameState,
    hand_index: int,
) -> PlaceCardAction:
    candidates = tuple(
        action
        for action in legal_actions(state)
        if isinstance(action, PlaceCardAction) and action.hand_index == hand_index
    )
    if not candidates:
        raise ValueError(f"selected hand slot is not placeable: {hand_index}")
    return min(candidates, key=lambda action: placement_sort_key(state, action))


def _current_hand_index(
    original_index: int,
    placed_original_indices: list[int],
) -> int:
    removed_before = sum(
        1 for placed_index in placed_original_indices if placed_index < original_index
    )
    return original_index - removed_before
