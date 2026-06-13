"""Optional Gymnasium wrapper for YellowstoneEnv."""

from __future__ import annotations

from typing import Any

from yellowstone.action_space import ACTION_SPACE_SIZE, legal_action_mask
from yellowstone.env import YellowstoneEnv
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.types import Phase

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ModuleNotFoundError:
    gym = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


_BaseEnv = gym.Env if gym is not None else object


class YellowstoneGymEnv(_BaseEnv):
    """Gymnasium-compatible wrapper around YellowstoneEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        player_count: int = 4,
        learning_player_index: int = 0,
    ) -> None:
        _require_gymnasium_dependencies()
        super().__init__()
        self.env = YellowstoneEnv(
            player_count=player_count,
            learning_player_index=learning_player_index,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(OBSERVATION_SIZE,),
            dtype=np.int16,
        )
        self._last_action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset using Gymnasium's reset signature."""
        del options
        super().reset(seed=seed)
        result = self.env.reset(seed=seed)
        observation = _observation_array(result.observation)
        info = dict(result.info)
        self._last_action_mask = _mask_array(result.legal_action_mask)
        info["action_mask"] = self._last_action_mask
        return observation, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step using Gymnasium's step signature."""
        result = self.env.step(int(action))
        observation = _observation_array(result.observation)
        info = dict(result.info)
        self._last_action_mask = _mask_array(result.legal_action_mask)
        info["action_mask"] = self._last_action_mask
        terminated = self.env.state is not None and self.env.state.phase == Phase.GAME_OVER
        truncated = bool(result.done and not terminated)
        return observation, result.reward, terminated, truncated, info

    def action_masks(self) -> Any:
        """Return legal action masks for sb3-contrib MaskablePPO."""
        if self.env.state is None:
            return self._last_action_mask
        self._last_action_mask = _mask_array(legal_action_mask(self.env.state))
        return self._last_action_mask


def gymnasium_available() -> bool:
    """Return whether optional Gymnasium dependencies are importable."""
    return gym is not None and np is not None and spaces is not None


def _require_gymnasium_dependencies() -> None:
    if not gymnasium_available():
        raise ImportError(
            "Gymnasium wrapper dependencies are not installed. "
            'Install them with: python -m pip install -e ".[rl]"'
        )


def _observation_array(observation: tuple[int, ...]) -> Any:
    _require_gymnasium_dependencies()
    return np.asarray(observation, dtype=np.int16)


def _mask_array(mask: tuple[bool, ...]) -> Any:
    _require_gymnasium_dependencies()
    return np.asarray(mask, dtype=np.bool_)
