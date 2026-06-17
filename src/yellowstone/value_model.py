"""Runtime helpers for learned state-value predictions."""

from __future__ import annotations

from functools import cache
from dataclasses import replace
from pathlib import Path
from typing import Any

from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.types import GameState, Phase
from yellowstone.value_training import make_state_value_model


def learned_state_loss_share(
    state: GameState,
    *,
    player_index: int,
    model_path: str,
    residual_by_hand_count: bool = False,
    allow_player_perspective: bool = False,
) -> float:
    """Predict final loss share for a learner turn-start state."""
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        return 0.0
    if state.current_player_index == player_index:
        evaluated_state = state
    elif allow_player_perspective:
        evaluated_state = replace(state, current_player_index=player_index)
    else:
        return 0.0
    observation = normalize_observation(state_to_observation(evaluated_state))
    predictor = _load_predictor(str(Path(model_path)))
    prediction, hand_count_baselines = predictor(observation)
    if not residual_by_hand_count:
        return prediction
    hand_count = len(state.players[player_index].hand)
    if 0 <= hand_count < len(hand_count_baselines):
        return prediction - hand_count_baselines[hand_count]
    return prediction


@cache
def _load_predictor(model_path: str) -> Any:
    torch = _import_torch()
    model = make_state_value_model(torch)
    payload = torch.load(model_path, map_location="cpu")
    if payload.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("state-value model observation size does not match")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    hand_count_baselines = tuple(
        float(value) for value in payload.get("hand_count_target_means", ())
    )

    def predict(observation: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
        with torch.no_grad():
            tensor = torch.tensor([observation], dtype=torch.float32)
            return float(model(tensor).item()), hand_count_baselines

    return predict


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for learned state-value reward. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch
