from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.turn_action_space import TURN_ACTION_SPACE_SIZE, legal_turn_action_indices
from yellowstone.turn_env import YellowstoneTurnEnv
from yellowstone.types import Phase


def test_turn_env_reset_returns_turn_action_mask() -> None:
    # turn-level環境のresetが36通りの行動maskを返すことを確認する。
    env = YellowstoneTurnEnv()

    result = env.reset(seed=1)

    assert len(result.observation) == OBSERVATION_SIZE
    assert len(result.legal_action_mask) == TURN_ACTION_SPACE_SIZE
    assert any(result.legal_action_mask)


def test_turn_env_step_applies_whole_learner_turn() -> None:
    # turn-level環境のstepが1手番分をまとめて進めることを確認する。
    env = YellowstoneTurnEnv()
    env.reset(seed=1)
    action_index = legal_turn_action_indices(env.state)[0]

    result = env.step(action_index)

    assert len(result.observation) == OBSERVATION_SIZE
    assert len(result.legal_action_mask) == TURN_ACTION_SPACE_SIZE
    assert result.done or result.info["current_player_index"] == 0
    assert env.state.phase in (Phase.PLAY, Phase.GAME_OVER)
