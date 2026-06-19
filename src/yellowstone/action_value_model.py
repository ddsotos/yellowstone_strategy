"""Runtime helpers for supervised action-value models."""

from __future__ import annotations

from functools import cache
from dataclasses import replace
from pathlib import Path
from typing import Any

from yellowstone.game import apply_known_legal_action
from yellowstone.action_value_training import (
    ACTION_VALUE_FEATURE_SIZE,
    make_action_value_model,
)
from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.turn_action_space import (
    TURN_ACTION_SPACE_SIZE,
    resolve_turn_action_before_refill,
)
from yellowstone.types import GameState


def predict_action_values(
    state: GameState,
    *,
    action_indices: tuple[int, ...],
    model_path: str | Path,
) -> tuple[float, ...]:
    """Predict action values for a turn-start state and candidate actions."""
    if not action_indices:
        return ()
    observation = normalize_observation(state_to_observation(state))
    after_observations = tuple(
        _after_decision_observation(state, action_index=action_index)
        for action_index in action_indices
    )
    predictor = _load_predictor(str(Path(model_path)))
    return predictor(observation, after_observations, action_indices)


@cache
def _load_predictor(model_path: str) -> Any:
    torch = _import_torch()
    model = make_action_value_model(torch)
    payload = torch.load(model_path, map_location="cpu")
    if payload.get("feature_size") != ACTION_VALUE_FEATURE_SIZE:
        raise ValueError("action-value model feature size does not match")
    if payload.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("action-value model observation size does not match")
    if payload.get("action_space_size") != TURN_ACTION_SPACE_SIZE:
        raise ValueError("action-value model action space size does not match")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    def predict(
        observation: tuple[float, ...],
        after_observations: tuple[tuple[float, ...], ...],
        action_indices: tuple[int, ...],
    ) -> tuple[float, ...]:
        features = [
            observation + after_observation + _action_one_hot(action_index)
            for action_index, after_observation in zip(
                action_indices,
                after_observations,
                strict=True,
            )
        ]
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32)
            values = model(tensor).flatten().tolist()
        return tuple(float(value) for value in values)

    return predict


def _after_decision_observation(
    state: GameState,
    *,
    action_index: int,
) -> tuple[float, ...]:
    next_state = state
    for action in resolve_turn_action_before_refill(state, action_index):
        next_state = apply_known_legal_action(next_state, action)
    learner_perspective = replace(next_state, current_player_index=state.current_player_index)
    return normalize_observation(state_to_observation(learner_perspective))


def _action_one_hot(action_index: int) -> tuple[float, ...]:
    values = [0.0] * TURN_ACTION_SPACE_SIZE
    values[action_index] = 1.0
    return tuple(values)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for action-value model evaluation. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch
