from yellowstone.rewards import reward_for_transition
from yellowstone.types import GameState, Phase, PlayerState


def test_reward_for_transition_rewards_loss_score_decrease() -> None:
    # 失点チャートが下がると正のrewardになることを確認する。
    before = GameState(
        players=(PlayerState(loss_score=5), PlayerState(), PlayerState(), PlayerState())
    )
    after = GameState(
        players=(PlayerState(loss_score=3), PlayerState(), PlayerState(), PlayerState())
    )

    assert reward_for_transition(before, after, player_index=0) == 2.0


def test_reward_for_transition_penalizes_negative_card_increase() -> None:
    # マイナスカードが増えると小さな負のrewardになることを確認する。
    before = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )
    after = GameState(
        players=(
            PlayerState(negative_cards=(object(), object())),  # type: ignore[arg-type]
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )

    assert reward_for_transition(before, after, player_index=0) == -0.2


def test_reward_for_transition_adds_terminal_reward() -> None:
    # ゲーム終了時は勝敗rewardが加算されることを確認する。
    before = GameState(
        players=(PlayerState(loss_score=5), PlayerState(), PlayerState(), PlayerState())
    )
    win_after = GameState(
        players=(PlayerState(loss_score=5), PlayerState(), PlayerState(), PlayerState()),
        phase=Phase.GAME_OVER,
        winners=(0,),
    )
    lose_after = GameState(
        players=(PlayerState(loss_score=5), PlayerState(), PlayerState(), PlayerState()),
        phase=Phase.GAME_OVER,
        winners=(1,),
    )

    assert reward_for_transition(before, win_after, player_index=0) == 1.0
    assert reward_for_transition(before, lose_after, player_index=0) == -1.0
