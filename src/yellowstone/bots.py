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


@dataclass(frozen=True, slots=True)
class _PlacementCandidate:
    action: PlaceCardAction
    state_after_action: GameState
    sort_key: tuple[int, ...]


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

    no_damage_second = _no_damage_candidates(
        state,
        place_actions,
        _player_negative_count(state),
    )
    if no_damage_second:
        return min(no_damage_second, key=lambda candidate: candidate.sort_key).action

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
    return _placement_sort_key_from_state(state, action, apply_action(state, action))


def _placement_sort_key_from_state(
    state: GameState,
    action: PlaceCardAction,
    state_after_action: GameState,
) -> tuple[int, ...]:
    player = state.players[state.current_player_index]
    card = player.hand[action.hand_index]
    return (
        _negative_card_delta_from_state(state, state_after_action),
        -_loss_score_delta_from_state(state, state_after_action),
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
    current_negative_count = _player_negative_count(state)
    first_candidates = _no_damage_candidates(
        state,
        first_actions,
        current_negative_count,
    )

    pairs: list[tuple[_PlacementCandidate, _PlacementCandidate]] = []
    for first_candidate in first_candidates:
        second_actions = _place_actions(legal_actions(first_candidate.state_after_action))
        second_candidates = _no_damage_candidates(
            first_candidate.state_after_action,
            second_actions,
            current_negative_count,
        )
        for second_candidate in second_candidates:
            pairs.append((first_candidate, second_candidate))
    if not pairs:
        return None

    def pair_key(
        pair: tuple[_PlacementCandidate, _PlacementCandidate],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        first_candidate, second_candidate = pair
        first_key = first_candidate.sort_key
        second_key = second_candidate.sort_key
        primary_key = min(first_key, second_key)
        secondary_key = second_key if primary_key == first_key else first_key
        return (primary_key, secondary_key, first_key, second_key)

    best_first, best_second = min(pairs, key=pair_key)
    return (best_first.action, best_second.action)


def _no_damage_candidates(
    state: GameState,
    actions: tuple[PlaceCardAction, ...],
    target_negative_count: int,
) -> tuple[_PlacementCandidate, ...]:
    candidates: list[_PlacementCandidate] = []
    for action in actions:
        next_state = apply_action(state, action)
        if _player_negative_count(next_state) != target_negative_count:
            continue
        candidates.append(
            _PlacementCandidate(
                action=action,
                state_after_action=next_state,
                sort_key=_placement_sort_key_from_state(state, action, next_state),
            )
        )
    return tuple(candidates)


def _negative_card_delta(state: GameState, action: PlaceCardAction) -> int:
    return _negative_card_delta_from_state(state, apply_action(state, action))


def _loss_score_delta(state: GameState, action: PlaceCardAction) -> int:
    return _loss_score_delta_from_state(state, apply_action(state, action))


def _negative_card_delta_from_state(
    state: GameState,
    state_after_action: GameState,
) -> int:
    return _player_negative_count(state_after_action) - _player_negative_count(state)


def _loss_score_delta_from_state(
    state: GameState,
    state_after_action: GameState,
) -> int:
    player_index = state.current_player_index
    before = state.players[player_index].loss_score
    after = state_after_action.players[player_index].loss_score
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
