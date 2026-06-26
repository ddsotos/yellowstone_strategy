"""Runtime inference for two-card rollout advantage regression."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_advantage_training import make_two_card_advantage_model


def predict_two_card_advantage_from_observation(
    observation: tuple[float, ...], model_path: str | Path
) -> float:
    """Predict stop loss minus two-card loss from a normalized observation."""
    return _load_predictor(str(Path(model_path)))(observation)


@cache
def _load_predictor(model_path: str) -> Any:
    torch = _import_torch()
    payload = torch.load(model_path, map_location="cpu")
    if payload.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("advantage model observation size does not match")
    model = make_two_card_advantage_model(torch)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    target_mean = float(payload["target_mean"])
    target_std = float(payload["target_std"])

    def predict(observation: tuple[float, ...]) -> float:
        with torch.no_grad():
            tensor = torch.tensor([observation], dtype=torch.float32)
            scaled = float(model(tensor).item())
            return scaled * target_std + target_mean

    return predict


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for advantage inference. "
            "Install the rl extra with: python -m pip install -e .[rl]"
        ) from exc
    return torch
