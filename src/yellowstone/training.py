"""Minimal MaskablePPO training entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yellowstone.gym_env import YellowstoneGymEnv, gymnasium_available

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
except ModuleNotFoundError:
    MaskablePPO = None  # type: ignore[assignment]
    MaskableEvalCallback = None  # type: ignore[assignment]


DEFAULT_MODEL_PATH = Path("models/yellowstone_maskable_ppo")
DEFAULT_LOG_DIR = Path("runs/yellowstone_maskable_ppo")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for a small local training run."""

    total_timesteps: int = 10_000
    seed: int = 0
    output_path: Path = DEFAULT_MODEL_PATH
    log_dir: Path = DEFAULT_LOG_DIR
    eval_freq: int = 5_000
    eval_episodes: int = 5
    learning_rate: float = 0.0003
    verbose: int = 1


def train_maskable_ppo(config: TrainingConfig) -> Path:
    """Train MaskablePPO with action masks and save the model."""
    _require_training_dependencies()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    env = YellowstoneGymEnv(normalize_observations=True)
    eval_env = YellowstoneGymEnv(normalize_observations=True)
    callback = _make_eval_callback(config, eval_env)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=config.seed,
        learning_rate=config.learning_rate,
        tensorboard_log=str(config.log_dir),
        verbose=config.verbose,
    )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        use_masking=True,
    )
    model.save(config.output_path)
    return config.output_path


def training_dependencies_available() -> bool:
    """Return whether optional RL training dependencies are importable."""
    return gymnasium_available() and MaskablePPO is not None and MaskableEvalCallback is not None


def _make_eval_callback(config: TrainingConfig, eval_env: YellowstoneGymEnv) -> Any:
    if config.eval_freq <= 0:
        return None
    best_model_dir = config.output_path.parent / "best"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    return MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(config.log_dir),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
    )


def _require_training_dependencies() -> None:
    if not training_dependencies_available():
        raise ImportError(
            "RL training dependencies are not installed. "
            'Install them with: python -m pip install -e ".[rl]"'
        )


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    return TrainingConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output_path=args.output_path,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
    )


def main() -> None:
    """Train a model from command-line arguments."""
    output_path = train_maskable_ppo(_parse_args())
    print(f"saved_model={output_path}")


if __name__ == "__main__":
    main()
