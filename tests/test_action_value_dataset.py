import pytest
from random import Random

from yellowstone.action_value_dataset import (
    ActionValueSample,
    _apply_rollout_action,
    collect_action_value_samples,
    read_action_value_samples,
    summarize_action_value_samples,
    write_action_value_samples,
)
from yellowstone.action_value_training import train_action_value_model
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.types import Card, Color, GameState, Phase, PlayerState
from yellowstone.types import RefillAction, RefillSource


def test_collect_action_value_samples_labels_turn_actions() -> None:
    # 手番開始状態と行動indexに、将来の有限horizon追加失点を付与できる。
    samples = collect_action_value_samples(
        source_games=1,
        source_state_limit=2,
        actions_per_state=2,
        horizon_learner_turns=2,
        source_seed_start=1,
    )

    assert samples
    assert all(len(sample.observation) == OBSERVATION_SIZE for sample in samples)
    assert all(0 <= sample.action_index < 36 for sample in samples)


def test_action_value_samples_round_trip_jsonl(tmp_path) -> None:
    # action-valueサンプルをJSONLへ保存し、同じ内容として読み戻せる。
    sample = ActionValueSample(
        observation=tuple(0.0 for _ in range(OBSERVATION_SIZE)),
        after_observation=tuple(0.1 for _ in range(OBSERVATION_SIZE)),
        action_index=6,
        target_self_loss=2.0,
        target_relative_loss=0.5,
        player_index=0,
        hand_count=6,
        seed=1,
        turn_index=0,
        heuristic_action_index=6,
        target_self_advantage=1.0,
        target_relative_advantage=0.5,
    )
    path = tmp_path / "action-value-samples.jsonl"

    write_action_value_samples((sample,), path)
    loaded = read_action_value_samples(path)

    assert loaded == (sample,)


def test_collect_action_value_samples_can_continue_past_game_end() -> None:
    # 継続ゲーム扱いなら、35点終了に遮られず長めhorizonの教師データを作れる。
    samples = collect_action_value_samples(
        source_games=1,
        source_state_limit=1,
        actions_per_state=1,
        horizon_learner_turns=20,
        source_seed_start=1,
        continuing_game=True,
        max_actions=20_000,
    )

    assert samples
    assert all(sample.target_self_loss >= 0 for sample in samples)


def test_continuing_deck_refill_keeps_hand_sorted() -> None:
    # 継続ゲームの山札切れ補充でも手札の数字昇順を保つ。
    state = GameState(
        players=(
            PlayerState(
                hand=(Card(Color.RED, 5), Card(Color.BLUE, 1)),
                negative_cards=(Card(Color.GREEN, 0),),
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=(Card(Color.YELLOW, 3),),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = _apply_rollout_action(
        state,
        RefillAction(RefillSource.DECK),
        rng=Random(1),
        continuing_game=True,
    )

    assert next_state.players[0].hand == (
        Card(Color.BLUE, 1),
        Card(Color.YELLOW, 3),
        Card(Color.RED, 5),
    )


def test_summarize_action_value_samples_counts_coverage() -> None:
    # action-valueサンプルの手札枚数と行動indexの分布を集計できる。
    samples = (
        ActionValueSample(
            tuple(0.0 for _ in range(OBSERVATION_SIZE)),
            tuple(0.1 for _ in range(OBSERVATION_SIZE)),
            6,
            2.0,
            0.5,
            0,
            6,
            1,
            0,
            6,
            0.0,
            0.0,
        ),
        ActionValueSample(
            tuple(0.0 for _ in range(OBSERVATION_SIZE)),
            tuple(0.2 for _ in range(OBSERVATION_SIZE)),
            7,
            1.0,
            -0.25,
            0,
            5,
            1,
            1,
            6,
            1.0,
            0.75,
        ),
    )

    summary = summarize_action_value_samples(
        samples,
        source_games=1,
        source_states=1,
        horizon_learner_turns=2,
        continuing_game=True,
    )

    assert summary.samples == 2
    assert summary.continuing_game is True
    assert summary.hand_count_histogram[6] == 1
    assert summary.action_count_histogram[6] == 1


def test_train_action_value_model_saves_model(tmp_path) -> None:
    # 最小データセットからaction-value modelを学習して保存できる。
    pytest.importorskip("torch")
    samples = (
        ActionValueSample(
            tuple(0.0 for _ in range(OBSERVATION_SIZE)),
            tuple(0.1 for _ in range(OBSERVATION_SIZE)),
            0,
            2.0,
            0.5,
            0,
            6,
            1,
            0,
            0,
            0.0,
            0.0,
        ),
        ActionValueSample(
            tuple(1.0 for _ in range(OBSERVATION_SIZE)),
            tuple(0.9 for _ in range(OBSERVATION_SIZE)),
            6,
            1.0,
            -0.25,
            0,
            6,
            1,
            1,
            0,
            1.0,
            0.75,
        ),
        ActionValueSample(
            tuple(0.5 for _ in range(OBSERVATION_SIZE)),
            tuple(0.6 for _ in range(OBSERVATION_SIZE)),
            7,
            3.0,
            1.0,
            0,
            5,
            1,
            2,
            0,
            -1.0,
            -0.5,
        ),
    )
    dataset_path = tmp_path / "action-value-samples.jsonl"
    model_path = tmp_path / "action-value-model.pt"

    write_action_value_samples(samples, dataset_path)
    result = train_action_value_model(
        dataset_path=dataset_path,
        output_path=model_path,
        epochs=1,
        batch_size=2,
        seed=1,
    )

    assert model_path.exists()
    assert result.sample_count == 3
    assert result.validation_count == 1
