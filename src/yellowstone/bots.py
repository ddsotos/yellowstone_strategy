"""Deterministic bot policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from yellowstone.game import apply_action, legal_actions
from yellowstone.types import (
    Action,
    Board,
    Card,
    Color,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)


class BotPolicy(Protocol):
    """Interface for deterministic or learned action policies."""

    def choose_action(self, state: GameState) -> Action | None:
        """Choose one legal action for the given state."""


@dataclass(frozen=True, slots=True)
class HeuristicBot:
    """Deterministic heuristic policy."""

    def choose_action(self, state: GameState) -> Action | None:
        """Choose an action using the documented heuristic rules."""
        return choose_heuristic_action(state)


def choose_heuristic_action(state: GameState) -> Action | None:
    """Choose one deterministic heuristic action."""
    actions = legal_actions(state)
    if not actions:
        return None
    if state.phase == Phase.GAME_OVER:
        return None
    if state.phase == Phase.REFILL:
        return _choose_refill_action(state, actions)
    if state.phase != Phase.PLAY:
        return None

    place_actions = _place_actions(actions)
    if not place_actions:
        end_turn = _first_action(actions, EndTurnAction)
        return end_turn

    if state.cards_played_this_turn == 0:
        no_damage_pair = _choose_no_damage_pair(state, place_actions)
        if no_damage_pair is not None:
            return no_damage_pair[0]
        return min(place_actions, key=lambda action: placement_sort_key(state, action))

    no_damage_second = _no_damage_actions(state, place_actions)
    if no_damage_second:
        return min(no_damage_second, key=lambda action: placement_sort_key(state, action))

    positive_actions = [
        action for action in place_actions if _loss_score_delta(state, action) > 0
    ]
    if positive_actions:
        return min(positive_actions, key=lambda action: placement_sort_key(state, action))

    if _first_action(actions, EndTurnAction) is not None:
        return _first_action(actions, EndTurnAction)
    return min(place_actions, key=lambda action: placement_sort_key(state, action))


def placement_sort_key(state: GameState, action: PlaceCardAction) -> tuple[int, ...]:
    """Return a deterministic sort key for one placement action.

    Smaller tuples are better. The order mirrors docs/heuristic-bot-design.md.
    """
    player = state.players[state.current_player_index]
    card = player.hand[action.hand_index]
    return (
        _negative_card_delta(state, action),
        -_loss_score_delta(state, action),
        -abs(card.rank_index - 3),
        0 if action.position in state.board else 1,
        -_same_color_remaining_count(player.hand, action.hand_index, card.color),
        -_board_color_count(state.board, card.color),
        action.hand_index,
        action.position.x,
        action.position.y,
        action.frame.x,
        action.frame.y,
    )


def _choose_refill_action(state: GameState, actions: tuple[Action, ...]) -> Action | None:
    player = state.players[state.current_player_index]
    if not player.hand and len(player.negative_cards) >= 6:
        negative_refill = RefillAction(RefillSource.NEGATIVE_CARDS)
        if negative_refill in actions:
            return negative_refill
    deck_refill = RefillAction(RefillSource.DECK)
    if deck_refill in actions:
        return deck_refill
    return min(actions, key=_action_tie_break_key)


def _choose_no_damage_pair(
    state: GameState,
    first_actions: tuple[PlaceCardAction, ...],
) -> tuple[PlaceCardAction, PlaceCardAction] | None:
    pairs: list[tuple[PlaceCardAction, PlaceCardAction, GameState]] = []
    for first_action in first_actions:
        first_state = apply_action(state, first_action)
        second_actions = _place_actions(legal_actions(first_state))
        for second_action in second_actions:
            second_state = apply_action(first_state, second_action)
            if _player_negative_count(second_state) == _player_negative_count(state):
                pairs.append((first_action, second_action, first_state))
    if not pairs:
        return None

    def pair_key(
        pair: tuple[PlaceCardAction, PlaceCardAction, GameState],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        first_action, second_action, first_state = pair
        first_key = placement_sort_key(state, first_action)
        second_key = placement_sort_key(first_state, second_action)
        primary_key = min(first_key, second_key)
        secondary_key = second_key if primary_key == first_key else first_key
        return (primary_key, secondary_key, first_key, second_key)

    best_first, best_second, _ = min(pairs, key=pair_key)
    return (best_first, best_second)


def _no_damage_actions(
    state: GameState,
    actions: tuple[PlaceCardAction, ...],
) -> tuple[PlaceCardAction, ...]:
    current_negative_count = _player_negative_count(state)
    return tuple(
        action
        for action in actions
        if _player_negative_count(apply_action(state, action)) == current_negative_count
    )


def _negative_card_delta(state: GameState, action: PlaceCardAction) -> int:
    return _player_negative_count(apply_action(state, action)) - _player_negative_count(state)


def _loss_score_delta(state: GameState, action: PlaceCardAction) -> int:
    player_index = state.current_player_index
    before = state.players[player_index].loss_score
    after = apply_action(state, action).players[player_index].loss_score
    return before - after


def _player_negative_count(state: GameState) -> int:
    return len(state.players[state.current_player_index].negative_cards)


def _same_color_remaining_count(
    hand: tuple[Card, ...],
    played_index: int,
    color: Color,
) -> int:
    return sum(
        1
        for index, card in enumerate(hand)
        if index != played_index and card.color == color
    )


def _board_color_count(board: Board, color: Color) -> int:
    return sum(
        1
        for stack in board.values()
        for card in stack
        if card.color == color
    )


def _place_actions(actions: tuple[Action, ...]) -> tuple[PlaceCardAction, ...]:
    return tuple(action for action in actions if isinstance(action, PlaceCardAction))


def _first_action[T: Action](
    actions: tuple[Action, ...],
    action_type: type[T],
) -> T | None:
    return next((action for action in actions if isinstance(action, action_type)), None)


def _action_tie_break_key(action: Action) -> tuple[int, ...]:
    if isinstance(action, PlaceCardAction):
        return (
            0,
            action.hand_index,
            action.position.x,
            action.position.y,
            action.frame.x,
            action.frame.y,
        )
    if isinstance(action, EndTurnAction):
        return (1,)
    if isinstance(action, RefillAction):
        refill_order = {
            RefillSource.NEGATIVE_CARDS: 0,
            RefillSource.DECK: 1,
            RefillSource.NONE: 2,
        }
        return (2, refill_order[action.source])
    return (3,)
