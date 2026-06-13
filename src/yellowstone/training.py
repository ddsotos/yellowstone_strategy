"""Minimal MaskablePPO training entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yellowstone.gym_env import YellowstoneTurnGymEnv, gymnasium_available

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
except ModuleNotFoundError:
    MaskablePPO = None  # type: ignore[assignment]
    MaskableEvalCallback = None  # type: ignore[assignment]
    CallbackList = None  # type: ignore[assignment]
    CheckpointCallback = None  # type: ignore[assignment]


DEFAULT_MODEL_PATH = Path("models/yellowstone_maskable_ppo")
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints")
DEFAULT_LOG_DIR = Path("runs/yellowstone_maskable_ppo")
DEFAULT_TRAINING_REPORT_PATH = Path("runs/yellowstone_maskable_ppo/training-report.json")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for a small local training run."""

    total_timesteps: int = 10_000
    seed: int = 0
    output_path: Path = DEFAULT_MODEL_PATH
    log_dir: Path = DEFAULT_LOG_DIR
    eval_freq: int = 5_000
    eval_episodes: int = 5
    checkpoint_freq: int = 0
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    resume_from: Path | None = None
    report_path: Path | None = DEFAULT_TRAINING_REPORT_PATH
    learning_rate: float = 0.0003
    verbose: int = 1


def train_maskable_ppo(config: TrainingConfig) -> Path:
    """Train MaskablePPO with action masks and save the model."""
    _require_training_dependencies()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = YellowstoneTurnGymEnv(normalize_observations=True)
    eval_env = YellowstoneTurnGymEnv(normalize_observations=True)
    callback = _make_callbacks(config, eval_env)
    model = _make_or_load_model(config, env)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        use_masking=True,
        reset_num_timesteps=config.resume_from is None,
    )
    model.save(config.output_path)
    if config.report_path is not None:
        write_training_report(config, saved_model_path=config.output_path)
    return config.output_path


def training_dependencies_available() -> bool:
    """Return whether optional RL training dependencies are importable."""
    return (
        gymnasium_available()
        and MaskablePPO is not None
        and MaskableEvalCallback is not None
        and CallbackList is not None
        and CheckpointCallback is not None
    )


def write_training_report(
    config: TrainingConfig,
    *,
    saved_model_path: Path,
) -> Path:
    """Write a JSON report describing one training run."""
    if config.report_path is None:
        raise ValueError("report_path is not configured")
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_model_path": str(saved_model_path),
        "config": _json_ready_config(config),
        "resume_enabled": config.resume_from is not None,
        "checkpoint_enabled": config.checkpoint_freq > 0,
    }
    config.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config.report_path


def _make_or_load_model(config: TrainingConfig, env: YellowstoneTurnGymEnv) -> Any:
    if config.resume_from is not None:
        model = MaskablePPO.load(config.resume_from)
        model.set_env(env)
        return model
    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=config.seed,
        learning_rate=config.learning_rate,
        tensorboard_log=str(config.log_dir),
        verbose=config.verbose,
    )


def _make_callbacks(config: TrainingConfig, eval_env: YellowstoneTurnGymEnv) -> Any:
    callbacks: list[Any] = []
    if config.eval_freq > 0:
        best_model_dir = config.output_path.parent / "best"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(config.log_dir),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.eval_episodes,
                deterministic=True,
            )
        )
    if config.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=str(config.checkpoint_dir),
                name_prefix="yellowstone_maskable_ppo",
            )
        )
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    return CallbackList(callbacks)


def _json_ready_config(config: TrainingConfig) -> dict[str, object]:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif value is None:
            values[key] = None
    return values


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
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_TRAINING_REPORT_PATH)
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
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        report_path=args.report_path,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
    )


def main() -> None:
    """Train a model from command-line arguments."""
    output_path = train_maskable_ppo(_parse_args())
    print(f"saved_model={output_path}")


if __name__ == "__main__":
    main()
