"""Reward helpers for reinforcement-learning environments."""

from __future__ import annotations

from yellowstone.types import GameState, Phase


NEGATIVE_CARD_REWARD_WEIGHT = 0.1
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
    if after.phase == Phase.GAME_OVER:
        reward += WIN_REWARD if player_index in after.winners else LOSS_REWARD
    return float(reward)
