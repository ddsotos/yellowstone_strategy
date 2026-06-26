"""Reusable heuristic turn plans for binary one-card/two-card decisions."""

from __future__ import annotations

from dataclasses import dataclass

from yellowstone.bots import HeuristicBot, placement_sort_key
from yellowstone.game import apply_known_legal_action, legal_actions
from yellowstone.types import Action, EndTurnAction, GameState, Phase, PlaceCardAction


@dataclass(frozen=True, slots=True)
class HeuristicTurnPlan:
    """Concrete heuristic actions and their immediate score bonus."""

    actions: tuple[Action, ...]
    bonus_score: int
    negative_card_delta: int


def choose_heuristic_one_card_plan(state: GameState) -> HeuristicTurnPlan | None:
    """Choose the standard heuristic first card and then end the turn."""
    if not _is_turn_start(state):
        return None
    first_action = HeuristicBot().choose_action(state)
    if not isinstance(first_action, PlaceCardAction):
        return None
    after_first = apply_known_legal_action(state, first_action)
    end_turn = EndTurnAction()
    if end_turn not in legal_actions(after_first):
        return None
    return HeuristicTurnPlan(
        actions=(first_action, end_turn),
        bonus_score=_loss_score_delta_between(state, after_first),
        negative_card_delta=_negative_card_delta_between(state, after_first),
    )


def choose_heuristic_two_card_plan(state: GameState) -> HeuristicTurnPlan | None:
    """Choose a two-card heuristic line from the original turn-start state."""
    if not _is_turn_start(state):
        return None
    best_candidate: tuple[
        tuple[int, ...], PlaceCardAction, PlaceCardAction, GameState, int, int
    ] | None = None
    initial_negative_count = len(
        state.players[state.current_player_index].negative_cards
    )
    initial_loss_score = state.players[state.current_player_index].loss_score
    for first_action in _place_actions(state):
        after_first = apply_known_legal_action(state, first_action)
        first_sort_key = placement_sort_key(state, first_action)
        for second_action in _place_actions(after_first):
            second_sort_key = placement_sort_key(after_first, second_action)
            second_negative_delta = second_sort_key[0]
            negative_delta = (
                len(
                    after_first.players[state.current_player_index].negative_cards
                )
                - initial_negative_count
                + second_negative_delta
            )
            second_score_gain = min(
                -second_sort_key[1],
                after_first.players[state.current_player_index].loss_score,
            )
            score_gain = (
                initial_loss_score
                - after_first.players[state.current_player_index].loss_score
                + second_score_gain
            )
            damage_priority = 0 if negative_delta == 0 else 1
            candidate_key = (
                damage_priority,
                negative_delta if damage_priority == 0 else -score_gain,
                0 if damage_priority == 0 else negative_delta,
                *first_sort_key,
                *second_sort_key,
            )
            if best_candidate is None or candidate_key < best_candidate[0]:
                best_candidate = (
                    candidate_key,
                    first_action,
                    second_action,
                    after_first,
                    score_gain,
                    negative_delta,
                )
    if best_candidate is None:
        return None
    _, first_action, second_action, after_first, score_gain, negative_delta = (
        best_candidate
    )
    after_second = apply_known_legal_action(after_first, second_action)
    return HeuristicTurnPlan(
        actions=(first_action, second_action),
        bonus_score=score_gain,
        negative_card_delta=negative_delta,
    )


def heuristic_turn_bonus_features(state: GameState) -> tuple[int, int]:
    """Return one-card and two-card heuristic bonus scores for observation."""
    features = heuristic_turn_features(state)
    return features[0], features[1]


def heuristic_turn_features(state: GameState) -> tuple[int, int, int, int]:
    """Return bonus and immediate-negative features for both turn plans."""
    one_card_plan = choose_heuristic_one_card_plan(state)
    two_card_plan = choose_heuristic_two_card_plan(state)
    return (
        0 if one_card_plan is None else one_card_plan.bonus_score,
        0 if two_card_plan is None else two_card_plan.bonus_score,
        0 if one_card_plan is None else one_card_plan.negative_card_delta,
        0 if two_card_plan is None else two_card_plan.negative_card_delta,
    )


def negative_card_delta_after_actions(
    state: GameState,
    actions: tuple[Action, ...],
) -> int:
    """Return current player's negative-card increase after concrete actions."""
    before = len(state.players[state.current_player_index].negative_cards)
    next_state = state
    for action in actions:
        next_state = apply_known_legal_action(next_state, action)
    after = len(next_state.players[state.current_player_index].negative_cards)
    return after - before


def _is_turn_start(state: GameState) -> bool:
    return state.phase == Phase.PLAY and state.cards_played_this_turn == 0


def _place_actions(state: GameState) -> tuple[PlaceCardAction, ...]:
    return tuple(
        action for action in legal_actions(state) if isinstance(action, PlaceCardAction)
    )


def _negative_card_delta_between(before: GameState, after: GameState) -> int:
    player_index = before.current_player_index
    return (
        len(after.players[player_index].negative_cards)
        - len(before.players[player_index].negative_cards)
    )


def _loss_score_delta_between(before: GameState, after: GameState) -> int:
    player_index = before.current_player_index
    return (
        before.players[player_index].loss_score
        - after.players[player_index].loss_score
    )
