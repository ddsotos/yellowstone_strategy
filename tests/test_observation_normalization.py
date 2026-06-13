from yellowstone.game import create_initial_state
from yellowstone.observation import OBSERVATION_SIZE, state_to_observation
from yellowstone.observation_normalization import (
    normalize_observation,
    observation_high_values,
    observation_normalization_policy,
)


def test_observation_high_values_match_observation_size() -> None:
    # 正規化用の上限値が観測ベクトルと同じ長さになることを確認する。
    highs = observation_high_values()

    assert len(highs) == OBSERVATION_SIZE
    assert all(value > 0 for value in highs)


def test_normalize_observation_scales_values_to_unit_range() -> None:
    # 生の整数観測を学習向けの0..1範囲へ変換できることを確認する。
    state = create_initial_state(4, seed=1)
    normalized = normalize_observation(state_to_observation(state))

    assert len(normalized) == OBSERVATION_SIZE
    assert all(0.0 <= value <= 1.0 for value in normalized)


def test_normalize_observation_rejects_wrong_size() -> None:
    # 観測サイズが違う場合に静かに誤変換しないことを確認する。
    try:
        normalize_observation((1, 2, 3))
    except ValueError as error:
        assert str(OBSERVATION_SIZE) in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_observation_normalization_policy_documents_core_choice() -> None:
    # core APIは整数のまま、Gym APIは正規化する方針を機械的に確認する。
    policy = observation_normalization_policy()

    assert policy["core_api"] == "raw_integer_tuple"
    assert policy["gym_api"] == "float32_normalized_by_default"
