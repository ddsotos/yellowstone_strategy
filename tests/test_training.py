from pathlib import Path

import pytest

from yellowstone.training import (
    TrainingConfig,
    train_maskable_ppo,
    training_dependencies_available,
    write_training_report,
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


def test_training_config_exposes_checkpoint_and_resume_controls() -> None:
    # 学習のcheckpoint、resume、report出力先を設定できることを確認する。
    config = TrainingConfig(
        checkpoint_freq=100,
        checkpoint_dir=Path("models/checkpoints-test"),
        resume_from=Path("models/old.zip"),
        report_path=Path("runs/report.json"),
    )

    assert config.checkpoint_freq == 100
    assert config.checkpoint_dir == Path("models/checkpoints-test")
    assert config.resume_from == Path("models/old.zip")
    assert config.report_path == Path("runs/report.json")


def test_write_training_report_outputs_json(tmp_path) -> None:
    # 学習設定と保存先をJSON reportとして残せることを確認する。
    report_path = tmp_path / "training-report.json"
    config = TrainingConfig(
        total_timesteps=10,
        output_path=tmp_path / "model",
        report_path=report_path,
    )

    written_path = write_training_report(
        config,
        saved_model_path=tmp_path / "model.zip",
    )

    content = written_path.read_text(encoding="utf-8")
    assert written_path == report_path
    assert '"total_timesteps": 10' in content
    assert "model.zip" in content


def test_train_maskable_ppo_reports_missing_optional_dependencies() -> None:
    # RL依存未導入環境では、学習実行時に導入方法つきで失敗することを確認する。
    if training_dependencies_available():
        pytest.skip("RL training dependencies are installed")

    with pytest.raises(ImportError, match=r'\.\[rl\]'):
        train_maskable_ppo(TrainingConfig(total_timesteps=1, verbose=0))
