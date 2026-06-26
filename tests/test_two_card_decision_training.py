import json

import pytest

from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    write_two_card_decision_samples,
)
from yellowstone.two_card_decision_training import train_two_card_decision_model


def test_binary_decision_model_trains_and_writes_report(tmp_path) -> None:
    # 二値ラベルの教師データからモデルと検証レポートを保存できる。
    pytest.importorskip("torch")
    samples = tuple(
        TwoCardDecisionSample(
            observation=tuple([float(label)] * OBSERVATION_SIZE),
            selected_decision=label,
            stop_self_loss=float(label),
            stop_relative_loss=float(label),
            two_card_self_loss=float(1 - label),
            two_card_relative_loss=float(1 - label),
            decision_case="test",
            player_index=0,
            hand_count=4,
            seed=index,
            turn_index=index,
        )
        for index, label in enumerate((0, 1, 0, 1, 0, 1, 0, 1))
    )
    dataset_path = tmp_path / "samples.jsonl"
    model_path = tmp_path / "model.pt"
    report_path = tmp_path / "report.json"
    write_two_card_decision_samples(samples, dataset_path)

    result = train_two_card_decision_model(
        dataset_path=dataset_path,
        output_path=model_path,
        report_path=report_path,
        epochs=2,
        batch_size=4,
        validation_ratio=0.25,
    )

    assert result.sample_count == 8
    assert model_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["validation_count"] == 2


def test_binary_decision_model_uses_separate_validation_dataset(tmp_path) -> None:
    # 独立seedの検証データを学習データへ混ぜずに評価できる。
    pytest.importorskip("torch")
    samples = tuple(
        TwoCardDecisionSample(
            observation=tuple([float(label)] * OBSERVATION_SIZE),
            selected_decision=label,
            stop_self_loss=float(label),
            stop_relative_loss=float(label),
            two_card_self_loss=float(1 - label),
            two_card_relative_loss=float(1 - label),
            decision_case="test",
            player_index=0,
            hand_count=4,
            seed=index,
            turn_index=index,
        )
        for index, label in enumerate((0, 1, 0, 1, 0, 1, 0, 1))
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_two_card_decision_samples(samples, train_path)
    write_two_card_decision_samples(samples[:4], validation_path)

    result = train_two_card_decision_model(
        dataset_path=train_path,
        validation_dataset_path=validation_path,
        train_sample_limit=6,
        output_path=tmp_path / "model.pt",
        epochs=1,
    )

    assert result.train_count == 6
    assert result.validation_count == 4
    assert result.validation_dataset_path == str(validation_path)
