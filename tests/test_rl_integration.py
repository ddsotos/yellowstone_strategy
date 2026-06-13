from dataclasses import replace

from yellowstone.action_space import action_to_index, legal_action_mask
from yellowstone.env import YellowstoneEnv
from yellowstone.game import apply_action, legal_actions
from yellowstone.observation import state_to_observation
from yellowstone.rewards import reward_for_transition
from yellowstone.serialization import game_state_from_dict
from yellowstone.types import Phase, PlaceCardAction


def test_env_reset_observation_and_mask_match_info_state() -> None:
    # reset結果の観測と合法手maskがinfo内の状態と一致することを確認する。
    env = YellowstoneEnv()

    result = env.reset(seed=1)
    state = game_state_from_dict(result.info["state"])

    assert result.observation == state_to_observation(state)
    assert result.legal_action_mask == legal_action_mask(state)


def test_env_step_reward_matches_transition_reward() -> None:
    # stepのrewardが遷移前後のGameStateから計算したrewardと一致することを確認する。
    env = YellowstoneEnv()
    env.reset(seed=1)
    before = env.state
    action = next(
        action for action in legal_actions(before) if isinstance(action, PlaceCardAction)
    )

    result = env.step(action_to_index(action))
    after = game_state_from_dict(result.info["state"])

    assert result.reward == reward_for_transition(before, after, player_index=0)


def test_env_step_mask_matches_next_decision_state() -> None:
    # step後の合法手maskが次の判断点の合法手と一致することを確認する。
    env = YellowstoneEnv()
    env.reset(seed=1)
    action = next(
        action
        for action in legal_actions(env.state)
        if isinstance(action, PlaceCardAction)
    )

    result = env.step(action_to_index(action))
    state = game_state_from_dict(result.info["state"])

    assert result.legal_action_mask == legal_action_mask(state)


def test_env_auto_advances_npcs_until_learning_player_or_done() -> None:
    # NPC手番は自動進行し、学習対象の手番または終了状態で止まることを確認する。
    env = YellowstoneEnv(learning_player_index=2)
    reset = env.reset(seed=1)

    assert reset.info["current_player_index"] == 2

    action = next(
        action
        for action in legal_actions(env.state)
        if isinstance(action, PlaceCardAction)
    )
    step = env.step(action_to_index(action))

    assert step.done or step.info["current_player_index"] == 2


def test_env_rejects_step_after_game_over() -> None:
    # ゲーム終了後のstepを拒否することを確認する。
    env = YellowstoneEnv()
    env.reset(seed=1)
    env.state = replace(env.state, phase=Phase.GAME_OVER)

    try:
        env.step(0)
    except RuntimeError as error:
        assert "cannot step after game over" in str(error)
    else:
        raise AssertionError("RuntimeError was not raised")
