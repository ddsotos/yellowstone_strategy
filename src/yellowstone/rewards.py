"""Reward helpers for reinforcement-learning environments."""

from __future__ import annotations

import os

from yellowstone.bots import placement_sort_key
from yellowstone.game import frame_positions, legal_actions
from yellowstone.turn_action_space import turn_action_from_index
from yellowstone.types import HAND_SIZE, GameState, Phase, PlaceCardAction


NEGATIVE_CARD_REWARD_WEIGHT = 0.1
DEFAULT_STATE_VALUE_REWARD_WEIGHT = 0.05
DEFAULT_NO_DAMAGE_TWO_CARD_TURN_MAX_REWARD = 0.0
NO_DAMAGE_CARD_VALUE = 1.0
NO_DAMAGE_TWO_CARD_PAIR_VALUE = 0.2
WIN_REWARD = 1.0
LOSS_REWARD = -1.0


def reward_for_transition(
    before: GameState,
    after: GameState,
    *,
    player_index: int,
) -> float:
    """Return the baseline reward for one environment transition."""
    before_player = before.players[player_index]
    after_player = after.players[player_index]
    loss_score_delta = before_player.loss_score - after_player.loss_score
    negative_card_delta = (
        len(before_player.negative_cards) - len(after_player.negative_cards)
    )
    reward = loss_score_delta + NEGATIVE_CARD_REWARD_WEIGHT * negative_card_delta
    reward += state_value_reward_weight() * (
        state_value(after, player_index=player_index)
        - state_value(before, player_index=player_index)
    )
    if after.phase == Phase.GAME_OVER:
        reward += WIN_REWARD if player_index in after.winners else LOSS_REWARD
    return float(reward)


def turn_action_reward(
    before: GameState,
    after_learner_turn: GameState,
    *,
    action_index: int,
    player_index: int,
) -> float:
    """Return an optional reward bonus for the selected turn-level action."""
    action = turn_action_from_index(action_index)
    if len(action.hand_indices) != 2:
        return 0.0
    before_count = len(before.players[player_index].negative_cards)
    after_count = len(after_learner_turn.players[player_index].negative_cards)
    if after_count > before_count:
        return 0.0
    before_hand_count = len(before.players[player_index].hand)
    draw_count = HAND_SIZE - max(0, before_hand_count - 2)
    draw_count = max(0, min(HAND_SIZE, draw_count))
    return no_damage_two_card_turn_max_reward() * draw_count / HAND_SIZE


def state_value(state: GameState, *, player_index: int) -> float:
    """Return a small hand-flexibility value for reward shaping."""
    if (
        state.phase != Phase.PLAY
        or state.current_player_index != player_index
        or state.cards_played_this_turn != 0
    ):
        return 0.0

    player = state.players[state.current_player_index]
    no_damage_card_count = 0
    for hand_index in range(min(len(player.hand), HAND_SIZE)):
        place_action = _choose_place_action_for_hand_index(state, hand_index)
        if _negative_card_delta(state, place_action) == 0:
            no_damage_card_count += 1

    no_damage_pair_count = no_damage_card_count * max(0, no_damage_card_count - 1)

    return (
        NO_DAMAGE_CARD_VALUE * no_damage_card_count
        + NO_DAMAGE_TWO_CARD_PAIR_VALUE * no_damage_pair_count
    )


def state_value_reward_weight() -> float:
    """Return the configured reward weight for hand-flexibility shaping."""
    configured = os.environ.get("YELLOWSTONE_STATE_VALUE_REWARD_WEIGHT")
    if configured is None:
        return DEFAULT_STATE_VALUE_REWARD_WEIGHT
    return float(configured)


def no_damage_two_card_turn_max_reward() -> float:
    """Return the max bonus for no-damage two-card turns with full refill value."""
    configured = os.environ.get("YELLOWSTONE_NO_DAMAGE_TWO_CARD_TURN_REWARD_WEIGHT")
    if configured is None:
        return DEFAULT_NO_DAMAGE_TWO_CARD_TURN_MAX_REWARD
    return float(configured)


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


def _negative_card_delta(state: GameState, action: PlaceCardAction) -> int:
    frame_cells = frame_positions(action.frame)
    return sum(
        len(stack)
        for position, stack in state.board.items()
        if position not in frame_cells
    )
