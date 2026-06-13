import pytest

from yellowstone.action_space import ACTION_SPACE_SIZE
from yellowstone.gym_env import YellowstoneGymEnv, gymnasium_available
from yellowstone.observation import OBSERVATION_SIZE


def test_gym_env_reports_missing_optional_dependencies() -> None:
    # Gymnasium未導入環境では明示的なImportErrorになることを確認する。
    if gymnasium_available():
        pytest.skip("Gymnasium dependencies are installed")

    with pytest.raises(ImportError, match=r'\.\[rl\]'):
        YellowstoneGymEnv()


def test_gym_env_reset_and_step_when_dependencies_are_available() -> None:
    # Gymnasium導入時にreset/step/action_masksがGymnasium形式で動くことを確認する。
    pytest.importorskip("gymnasium")
    pytest.importorskip("numpy")
    from yellowstone.game import legal_actions
    from yellowstone.action_space import action_to_index
    from yellowstone.types import PlaceCardAction

    env = YellowstoneGymEnv()
    observation, info = env.reset(seed=1)
    action = next(
        action
        for action in legal_actions(env.env.state)
        if isinstance(action, PlaceCardAction)
    )

    step = env.step(action_to_index(action))

    assert observation.shape == (OBSERVATION_SIZE,)
    assert str(observation.dtype) == "float32"
    assert env.action_space.n == ACTION_SPACE_SIZE
    assert str(env.observation_space.dtype) == "float32"
    assert info["action_mask"].shape == (ACTION_SPACE_SIZE,)
    assert len(step) == 5
    assert env.action_masks().shape == (ACTION_SPACE_SIZE,)


def test_gym_env_can_return_raw_integer_observations_when_requested() -> None:
    # 必要な場合はGymnasium wrapperでも未正規化の整数観測を返せることを確認する。
    pytest.importorskip("gymnasium")
    pytest.importorskip("numpy")

    env = YellowstoneGymEnv(normalize_observations=False)
    observation, _ = env.reset(seed=1)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert str(observation.dtype) == "int16"
