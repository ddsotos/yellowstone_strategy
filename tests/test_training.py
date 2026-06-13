from pathlib import Path

import pytest

from yellowstone.training import (
    TrainingConfig,
    train_maskable_ppo,
    training_dependencies_available,
)


def test_training_config_exposes_minimal_run_controls() -> None:
    # 学習スクリプトがseed、step数、保存先、評価間隔を設定できることを確認する。
    config = TrainingConfig(
        total_timesteps=123,
        seed=7,
        output_path=Path("models/test_model"),
        eval_freq=50,
    )

    assert config.total_timesteps == 123
    assert config.seed == 7
    assert config.output_path == Path("models/test_model")
    assert config.eval_freq == 50


def test_train_maskable_ppo_reports_missing_optional_dependencies() -> None:
    # RL依存未導入環境では、学習実行時に導入方法つきで失敗することを確認する。
    if training_dependencies_available():
        pytest.skip("RL training dependencies are installed")

    with pytest.raises(ImportError, match=r'\.\[rl\]'):
        train_maskable_ppo(TrainingConfig(total_timesteps=1, verbose=0))
