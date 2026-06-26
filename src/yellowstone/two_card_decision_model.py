"""Runtime helpers for the learned one-card/two-card decision."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.two_card_decision_training import make_two_card_decision_model
from yellowstone.types import GameState


def predict_two_card_probability(state: GameState, model_path: str | Path) -> float:
    """Predict the probability of choosing the two-card heuristic plan."""
    observation = normalize_observation(state_to_observation(state))
    return predict_two_card_probability_from_observation(observation, model_path)


def predict_two_card_probability_from_observation(
    observation: tuple[float, ...], model_path: str | Path
) -> float:
    """Predict from an already normalized observation."""
    predictor = _load_predictor(str(Path(model_path)))
    return predictor(observation)


@cache
def _load_predictor(model_path: str) -> Any:
    torch = _import_torch()
    payload = torch.load(model_path, map_location="cpu")
    if payload.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("binary decision model observation size does not match")
    model = make_two_card_decision_model(torch)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    def predict(observation: tuple[float, ...]) -> float:
        with torch.no_grad():
            tensor = torch.tensor([observation], dtype=torch.float32)
            return float(torch.sigmoid(model(tensor)).item())

    return predict


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for binary decision inference. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch
