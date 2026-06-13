from random import Random

from yellowstone.cli import (
    TurnLog,
    create_game_log,
    play_current_player_turn,
    render_action,
    render_current_hand,
    render_observation,
    render_turn_log,
)
from yellowstone.types import (
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    PlayerState,
    Position,
    RefillAction,
    RefillSource,
)


class ScriptedBot:
    def __init__(self, actions):
        self._actions = list(actions)

    def choose_action(self, _state):
        if not self._actions:
            return None
        return self._actions.pop(0)


def test_play_current_player_turn_stops_at_next_player() -> None:
    # 1回のEnterで現在プレイヤーの手番だけをまとめて進めることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    bot = ScriptedBot(
        [
            PlaceCardAction(
                hand_index=0,
                position=Position(0, 0),
                frame=Frame(0, 0),
            ),
            EndTurnAction(),
        ]
    )

    next_state, actions = play_current_player_turn(state, bot, rng=Random(1))

    assert next_state.current_player_index == 1
    assert actions == (
        "place hand[0]=R1 at (0,0) frame=(0,0)",
        "end turn after one card",
    )


def test_create_game_log_records_completed_turns() -> None:
    # 先にゲームを進めてから閲覧用の手番ログを作れることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    bot = ScriptedBot(
        [
            PlaceCardAction(
                hand_index=0,
                position=Position(0, 0),
                frame=Frame(0, 0),
            ),
            EndTurnAction(),
        ]
    )

    next_state, logs = create_game_log(state, bot, rng=Random(1))

    assert next_state.current_player_index == 1
    assert len(logs) == 1
    assert logs[0].turn_number == 1
    assert logs[0].player_index == 0
    assert logs[0].actions == (
        "place hand[0]=R1 at (0,0) frame=(0,0)",
        "end turn after one card",
    )
    assert "phase=play current=P1" in logs[0].observation_after_turn


def test_render_turn_log_includes_actions_and_observation() -> None:
    # 閲覧用ログに手番の行動一覧と手番後の状態が含まれることを確認する。
    turn_log = TurnLog(
        turn_number=2,
        player_index=1,
        actions=("refill source=deck",),
        observation_after_turn="phase=play current=P2",
    )

    rendered = render_turn_log(turn_log)

    assert "Turn 2: P1" in rendered
    assert "  1. refill source=deck" in rendered
    assert rendered.endswith("phase=play current=P2")


def test_render_current_hand_shows_indexes_and_cards() -> None:
    # 現在プレイヤーの手札をindex付きで確認できることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0), Card(Color.BLUE, 6))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )

    assert render_current_hand(state) == "P0 hand: 0:R1 1:B7"


def test_render_action_shows_place_details() -> None:
    # 配置行動のカード、座標、枠を表示できることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.GREEN, 2),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    action = PlaceCardAction(
        hand_index=0,
        position=Position(3, 2),
        frame=Frame(2, 0),
    )

    assert render_action(state, action) == "place hand[0]=G3 at (3,2) frame=(2,0)"


def test_render_action_shows_non_place_actions() -> None:
    # 終了と補充の行動も読める文字列になることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    assert render_action(state, EndTurnAction()) == "end turn after one card"
    assert (
        render_action(state, RefillAction(RefillSource.NEGATIVE_CARDS))
        == "refill source=negative_cards"
    )


def test_render_observation_includes_board_summary_and_hand() -> None:
    # NPC確認画面に盤面、プレイヤー概要、現在手札が含まれることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 0): (Card(Color.RED, 0),)},
    )

    rendered = render_observation(state)

    assert "phase=play current=P0" in rendered
    assert "R1" in rendered
    assert "*P0: hand=1 negative=0 loss=5" in rendered
    assert "P0 hand: 0:R1" in rendered


def test_render_observation_shows_winners_on_game_over() -> None:
    # ゲーム終了時は手札ではなく勝者を表示することを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        phase=Phase.GAME_OVER,
        winners=(2,),
    )

    assert "winners=2" in render_observation(state)
