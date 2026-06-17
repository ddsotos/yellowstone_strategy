import pytest

from yellowstone.rewards import (
    learned_state_value_reward,
    learned_state_value_model_path,
    learned_state_value_residual_by_hand_count,
    learned_state_value_reward_weight,
    reward_for_transition,
    state_value,
    state_value_reward_weight,
    turn_action_reward,
    two_card_turn_max_reward,
)
from yellowstone.types import Card, Color, GameState, Phase, PlayerState, Position


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


def test_state_value_counts_no_damage_turn_options() -> None:
    # 失点なしで置けるカードと2枚ペアがある状態を高く評価する。
    red_one = Card(Color.RED, 0)
    red_two = Card(Color.RED, 1)
    state = GameState(
        players=(
            PlayerState(hand=(red_one, red_two)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 0): (red_one,)},
    )

    assert state_value(state, player_index=0) == 2.4


def test_reward_for_transition_penalizes_reduced_state_value() -> None:
    # 即時失点がなくても次に失点なしで動ける余地が減ると小さな負報酬になる。
    red_one = Card(Color.RED, 0)
    red_two = Card(Color.RED, 1)
    before = GameState(
        players=(
            PlayerState(hand=(red_one, red_two)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 0): (red_one,)},
    )
    after = GameState(
        players=(
            PlayerState(hand=(red_one,)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 0): (red_one,)},
    )

    assert reward_for_transition(before, after, player_index=0) == pytest.approx(-0.07)


def test_state_value_reward_weight_can_be_configured(monkeypatch) -> None:
    # 状態評価rewardの重みを実験ごとに切り替えられる。
    monkeypatch.setenv("YELLOWSTONE_STATE_VALUE_REWARD_WEIGHT", "0.2")

    assert state_value_reward_weight() == 0.2


def test_two_card_turn_reward_is_configurable(monkeypatch) -> None:
    # 2枚ターンの最大追加rewardを実験ごとに切り替えられる。
    monkeypatch.setenv("YELLOWSTONE_TWO_CARD_TURN_REWARD_WEIGHT", "0.5")

    assert two_card_turn_max_reward() == 0.5


def test_learned_state_value_reward_is_configurable(monkeypatch) -> None:
    # 学習済み状態価値rewardのモデルパスと重みを環境変数で切り替えられる。
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_MODEL_PATH", "model.pt")
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_REWARD_WEIGHT", "0.7")
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_RESIDUAL_BY_HAND", "1")

    assert learned_state_value_model_path() == "model.pt"
    assert learned_state_value_reward_weight() == 0.7
    assert learned_state_value_residual_by_hand_count()


def test_reward_for_transition_adds_learned_loss_share_improvement(
    monkeypatch,
) -> None:
    # 予測失点割合が下がる状態遷移は、学習済み状態価値rewardで正に評価される。
    before = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    after = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 1),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    monkeypatch.setenv("YELLOWSTONE_STATE_VALUE_REWARD_WEIGHT", "0.0")
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_MODEL_PATH", "model.pt")
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_REWARD_WEIGHT", "2.0")

    def fake_loss_share(
        state,
        *,
        player_index,
        model_path,
        residual_by_hand_count,
        allow_player_perspective=False,
    ):
        assert player_index == 0
        assert model_path == "model.pt"
        assert not residual_by_hand_count
        assert not allow_player_perspective
        return 0.30 if state is before else 0.20

    monkeypatch.setattr("yellowstone.rewards.learned_state_loss_share", fake_loss_share)

    assert reward_for_transition(before, after, player_index=0) == pytest.approx(0.2)


def test_learned_state_value_reward_can_use_post_refill_state(
    monkeypatch,
) -> None:
    # turn-level学習ではNPC手番後ではなく、補充後のP0視点状態で学習済み状態価値を評価する。
    before = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0), Card(Color.RED, 1))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    after_refill = GameState(
        players=(
            PlayerState(hand=tuple(Card(Color.RED, index) for index in range(6))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        current_player_index=1,
    )
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_MODEL_PATH", "model.pt")
    monkeypatch.setenv("YELLOWSTONE_LEARNED_STATE_VALUE_REWARD_WEIGHT", "2.0")

    def fake_loss_share(
        state,
        *,
        player_index,
        model_path,
        residual_by_hand_count,
        allow_player_perspective=False,
    ):
        assert player_index == 0
        assert model_path == "model.pt"
        if state is after_refill:
            assert allow_player_perspective
            return 0.15
        return 0.25

    monkeypatch.setattr("yellowstone.rewards.learned_state_loss_share", fake_loss_share)

    assert learned_state_value_reward(
        before,
        after_refill,
        player_index=0,
        allow_after_player_perspective=True,
    ) == pytest.approx(0.2)


def test_turn_action_reward_scales_by_draw_count(monkeypatch) -> None:
    # 2枚ターンの追加rewardが、補充で引ける枚数に比例する。
    monkeypatch.setenv("YELLOWSTONE_TWO_CARD_TURN_REWARD_WEIGHT", "1.8")
    cards = tuple(Card(Color.RED, index) for index in range(6))
    before = GameState(
        players=(
            PlayerState(hand=cards),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    after = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    assert turn_action_reward(before, after, action_index=6, player_index=0) == 0.6
    assert turn_action_reward(before, after, action_index=0, player_index=0) == 0.0


def test_turn_action_reward_is_larger_with_low_hand_count(monkeypatch) -> None:
    # 手札が少ないほど2枚出し後に多く補充できるため、追加rewardが大きくなる。
    monkeypatch.setenv("YELLOWSTONE_TWO_CARD_TURN_REWARD_WEIGHT", "1.8")
    cards = (Card(Color.RED, 0), Card(Color.RED, 1))
    before = GameState(
        players=(
            PlayerState(hand=cards),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    after = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    assert turn_action_reward(before, after, action_index=6, player_index=0) == 1.8


def test_turn_action_reward_allows_negative_card_increase(monkeypatch) -> None:
    # マイナスカードが増える2枚出しでも、補充価値の追加rewardは付く。
    monkeypatch.setenv("YELLOWSTONE_TWO_CARD_TURN_REWARD_WEIGHT", "1.8")
    cards = tuple(Card(Color.RED, index) for index in range(6))
    before = GameState(
        players=(
            PlayerState(hand=cards, negative_cards=()),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    after = GameState(
        players=(
            PlayerState(negative_cards=(Card(Color.BLUE, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )

    assert turn_action_reward(before, after, action_index=6, player_index=0) == 0.6
