from random import Random

import pytest

from yellowstone.bots import ExploratoryHeuristicBot
from yellowstone.game import apply_known_legal_action
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.types import Card, Color, GameState, PlayerState
from yellowstone.value_dataset import (
    StateValueSample,
    collect_state_value_samples,
    read_state_value_samples,
    summarize_state_value_samples,
    write_state_value_samples,
)
from yellowstone.value_counterfactual_dataset import (
    collect_counterfactual_state_value_samples,
)
from yellowstone.value_training import train_state_value_model


def test_exploratory_heuristic_can_stop_after_one_no_damage_card() -> None:
    # 探索用heuristicは、2枚を失点なしで出せる場面でも確率設定により1枚で止められる。
    bot = ExploratoryHeuristicBot(
        rng=Random(1),
        one_card_probabilities={2: 1.0},
    )
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0), Card(Color.BLUE, 1))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )

    first_action = bot.choose_action(state)
    assert first_action is not None
    after_first = apply_known_legal_action(state, first_action)
    second_action = bot.choose_action(after_first)

    assert second_action is not None
    assert second_action.__class__.__name__ == "EndTurnAction"


def test_collect_state_value_samples_labels_turn_start_states() -> None:
    # heuristicロールアウトから手番開始状態と最終失点割合ラベルを収集できることを確認する。
    samples = collect_state_value_samples(games=1, seed_start=1)

    assert samples
    assert all(len(sample.observation) == OBSERVATION_SIZE for sample in samples)
    assert all(0.0 <= sample.target_loss_share <= 1.0 for sample in samples)
    assert all(0 <= sample.hand_count <= 6 for sample in samples)


def test_collect_counterfactual_state_value_samples_from_turn_actions() -> None:
    # 学習者の候補turn action後の状態を、将来失点割合ラベル付きで収集できる。
    samples = collect_counterfactual_state_value_samples(
        source_games=1,
        source_state_limit=2,
        actions_per_state=2,
        source_seed_start=1,
    )

    assert samples
    assert all(sample.player_index == 0 for sample in samples)
    assert all(len(sample.observation) == OBSERVATION_SIZE for sample in samples)
    assert all(0.0 <= sample.target_loss_share <= 1.0 for sample in samples)


def test_state_value_samples_round_trip_jsonl(tmp_path) -> None:
    # 状態価値サンプルをJSONLへ保存し、同じ内容として読み戻せることを確認する。
    sample = StateValueSample(
        observation=tuple(0.0 for _ in range(OBSERVATION_SIZE)),
        target_loss_share=0.25,
        player_index=0,
        hand_count=6,
        seed=1,
        turn_index=0,
    )
    path = tmp_path / "value-samples.jsonl"

    write_state_value_samples((sample,), path)
    loaded = read_state_value_samples(path)

    assert loaded == (sample,)


def test_summarize_state_value_samples_counts_hand_histogram() -> None:
    # 状態価値サンプルの手札枚数別件数を集計できることを確認する。
    samples = (
        StateValueSample(tuple(0.0 for _ in range(OBSERVATION_SIZE)), 0.2, 0, 6, 1, 0),
        StateValueSample(tuple(0.0 for _ in range(OBSERVATION_SIZE)), 0.3, 1, 4, 1, 1),
    )

    summary = summarize_state_value_samples(samples, games=1, completed_games=1)

    assert summary.samples == 2
    assert summary.hand_count_histogram[6] == 1
    assert summary.hand_count_histogram[4] == 1


def test_train_state_value_model_saves_model(tmp_path) -> None:
    # 最小データセットから教師ありvalue modelを学習し、保存できることを確認する。
    pytest.importorskip("torch")
    samples = (
        StateValueSample(tuple(0.0 for _ in range(OBSERVATION_SIZE)), 0.2, 0, 6, 1, 0),
        StateValueSample(tuple(1.0 for _ in range(OBSERVATION_SIZE)), 0.4, 1, 4, 1, 1),
        StateValueSample(tuple(0.5 for _ in range(OBSERVATION_SIZE)), 0.3, 2, 5, 1, 2),
    )
    dataset_path = tmp_path / "value-samples.jsonl"
    model_path = tmp_path / "value-model.pt"

    write_state_value_samples(samples, dataset_path)
    result = train_state_value_model(
        dataset_path=dataset_path,
        output_path=model_path,
        epochs=1,
        batch_size=2,
        seed=1,
    )

    assert model_path.exists()
    assert result.sample_count == 3
    assert result.validation_count == 1
