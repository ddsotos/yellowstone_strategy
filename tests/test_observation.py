from yellowstone.observation import (
    BOARD_OBSERVATION_SIZE,
    HAND_OBSERVATION_SIZE,
    OBSERVATION_SIZE,
    PLAYERS_OBSERVATION_SIZE,
    observation_metadata,
    state_to_observation,
)
from yellowstone.types import (
    Card,
    Color,
    GameState,
    PlayerState,
    Position,
)


def test_state_to_observation_has_fixed_size() -> None:
    # 観測が常に固定長になることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    observation = state_to_observation(state)

    assert len(observation) == OBSERVATION_SIZE
    assert observation_metadata()["observation_size"] == OBSERVATION_SIZE


def test_state_to_observation_encodes_compressed_board_counts() -> None:
    # 盤面は3x3枠の基準位置とセルごとの総枚数だけで入ることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        board={
            Position(1, 2): (
                Card(Color.RED, 2),
                Card(Color.BLUE, 2),
                Card(Color.RED, 2),
            )
        },
    )

    observation = state_to_observation(state)

    assert observation[:BOARD_OBSERVATION_SIZE] == (
        2,
        1,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_state_to_observation_encodes_relative_hand_colors_from_board_columns() -> None:
    # 手札の色は絶対色ではなく、盤面3列に対する相対色one-hotで入ることを確認する。
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.RED, 0),
                    Card(Color.BLUE, 1),
                    Card(Color.GREEN, 2),
                    Card(Color.YELLOW, 3),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(1, 2): (Card(Color.GREEN, 2),),
            Position(2, 2): (Card(Color.BLUE, 2),),
            Position(3, 2): (Card(Color.RED, 2),),
        },
    )

    observation = state_to_observation(state)
    hand_start = BOARD_OBSERVATION_SIZE

    assert observation[hand_start : hand_start + 24] == (
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        2,
        1,
        0,
        0,
        0,
        1,
        3,
    )


def test_state_to_observation_assigns_unseen_hand_colors_without_absolute_order() -> None:
    # 盤面にない色は手札スロット順で相対色IDへ割り当てられることを確認する。
    state = GameState(
        players=(
            PlayerState(),
            PlayerState(hand=(Card(Color.GREEN, 4),)),
            PlayerState(),
            PlayerState(),
        ),
        current_player_index=1,
    )

    observation = state_to_observation(state)
    hand_start = BOARD_OBSERVATION_SIZE

    assert observation[hand_start : hand_start + 6] == (1, 1, 0, 0, 0, 4)
    assert observation[hand_start + 6 : hand_start + 12] == (0, 0, 0, 0, 0, 0)


def test_state_to_observation_encodes_players_and_scalars() -> None:
    # プレイヤー状態と、山札段階・決算回数だけが末尾に入ることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),), loss_score=5),
            PlayerState(negative_cards=(Card(Color.BLUE, 1),), loss_score=8),
            PlayerState(),
            PlayerState(hand=(Card(Color.GREEN, 1), Card(Color.YELLOW, 2))),
        ),
        deck=tuple(Card(Color.YELLOW, index % 7) for index in range(7)),
        settlement_count=3,
    )

    observation = state_to_observation(state)
    player_start = BOARD_OBSERVATION_SIZE + HAND_OBSERVATION_SIZE
    scalar_start = player_start + PLAYERS_OBSERVATION_SIZE

    assert observation[player_start : player_start + 12] == (
        1,
        0,
        5,
        0,
        1,
        8,
        0,
        0,
        5,
        2,
        0,
        5,
    )
    assert observation[scalar_start : scalar_start + 2] == (2, 3)
