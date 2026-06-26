import json

import pytest

from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.two_card_advantage_training import train_two_card_advantage_model
from yellowstone.two_card_decision_dataset import (
    TwoCardDecisionSample,
    write_two_card_decision_samples,
)


def test_advantage_regression_trains_with_fixed_validation(tmp_path) -> None:
    # 損失差回帰を独立validationで学習し、指標とモデルを保存できる。
    pytest.importorskip("torch")
    samples = tuple(
        TwoCardDecisionSample(
            observation=tuple(float(index % 2) for _ in range(OBSERVATION_SIZE)),
            selected_decision=index % 2,
            stop_self_loss=float(index % 3),
            stop_relative_loss=float(index - 4),
            two_card_self_loss=float((index + 1) % 3),
            two_card_relative_loss=0.0,
            decision_case="heuristic_stops_with_second_available",
            player_index=0,
            hand_count=4,
            seed=index,
            turn_index=index,
        )
        for index in range(8)
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    report_path = tmp_path / "report.json"
    model_path = tmp_path / "model.pt"
    write_two_card_decision_samples(samples, train_path)
    write_two_card_decision_samples(samples[:4], validation_path)

    result = train_two_card_advantage_model(
        dataset_path=train_path,
        validation_dataset_path=validation_path,
        output_path=model_path,
        report_path=report_path,
        epochs=2,
    )

    assert result.train_count == 8
    assert result.validation_count == 4
    assert model_path.exists()
    assert "validation_mae" in json.loads(report_path.read_text(encoding="utf-8"))


def test_advantage_regression_restores_early_stopping_epoch(tmp_path) -> None:
    # validation改善が止まったら早期終了し、最良epochを結果へ残す。
    pytest.importorskip("torch")
    samples = tuple(
        TwoCardDecisionSample(
            observation=tuple(float(index % 2) for _ in range(OBSERVATION_SIZE)),
            selected_decision=index % 2,
            stop_self_loss=float(index),
            stop_relative_loss=float(index),
            two_card_self_loss=0.0,
            two_card_relative_loss=0.0,
            decision_case="heuristic_stops_with_second_available",
            player_index=0,
            hand_count=4,
            seed=index,
            turn_index=index,
        )
        for index in range(8)
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_two_card_decision_samples(samples, train_path)
    write_two_card_decision_samples(samples[:4], validation_path)

    result = train_two_card_advantage_model(
        dataset_path=train_path,
        validation_dataset_path=validation_path,
        output_path=tmp_path / "model.pt",
        epochs=20,
        early_stopping_patience=1,
        early_stopping_min_delta=1_000_000.0,
    )

    assert result.best_epoch == 1
    assert result.epochs_trained == 2


def test_advantage_regression_combines_training_datasets(tmp_path) -> None:
    # 複数seed範囲の教師JSONLを複製せず一つの学習へ投入できる。
    pytest.importorskip("torch")
    samples = tuple(
        TwoCardDecisionSample(
            observation=tuple(float(index % 2) for _ in range(OBSERVATION_SIZE)),
            selected_decision=index % 2,
            stop_self_loss=float(index),
            stop_relative_loss=float(index),
            two_card_self_loss=0.0,
            two_card_relative_loss=0.0,
            decision_case="heuristic_stops_with_second_available",
            player_index=0,
            hand_count=4,
            seed=index,
            turn_index=index,
        )
        for index in range(6)
    )
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_two_card_decision_samples(samples[:3], first_path)
    write_two_card_decision_samples(samples[3:], second_path)
    write_two_card_decision_samples(samples[:2], validation_path)

    result = train_two_card_advantage_model(
        dataset_path=first_path,
        additional_dataset_paths=(second_path,),
        validation_dataset_path=validation_path,
        output_path=tmp_path / "model.pt",
        epochs=1,
    )

    assert result.train_count == 6


def test_advantage_regression_filters_decision_cases(tmp_path) -> None:
    # 謇矩・′1譫壹〒豁｢縺ｾ繧句ｱ髱｢縺縺代ｒ謚懈爾縺励※蟄ｦ鄙偵〒縺阪ｋ縲・    pytest.importorskip("torch")
    samples = tuple(
        TwoCardDecisionSample(
            observation=tuple(float(index % 2) for _ in range(OBSERVATION_SIZE)),
            selected_decision=index % 2,
            stop_self_loss=float(index),
            stop_relative_loss=float(index),
            two_card_self_loss=0.0,
            two_card_relative_loss=0.0,
            decision_case=(
                "heuristic_stops_with_second_available"
                if index < 4
                else "heuristic_plays_damaging_second"
            ),
            player_index=0,
            hand_count=4,
            seed=index,
            turn_index=index,
        )
        for index in range(8)
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_two_card_decision_samples(samples, train_path)
    write_two_card_decision_samples(samples, validation_path)

    result = train_two_card_advantage_model(
        dataset_path=train_path,
        validation_dataset_path=validation_path,
        output_path=tmp_path / "model.pt",
        epochs=1,
        decision_case_filter=("heuristic_stops_with_second_available",),
    )

    assert result.train_count == 4
    assert result.validation_count == 4
    assert result.decision_case_filter == ("heuristic_stops_with_second_available",)
