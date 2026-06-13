from yellowstone.observation import (
    BOARD_OBSERVATION_SIZE,
    HAND_OBSERVATION_SIZE,
    OBSERVATION_SIZE,
    observation_metadata,
    state_to_observation,
)
from yellowstone.types import (
    Card,
    Color,
    GameState,
    Phase,
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


def test_state_to_observation_encodes_board_stack_color_counts() -> None:
    # 盤面セルに重ね置きされたカードの色枚数が入ることを確認する。
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
    cell_start = ((2 * 7) + 1) * 4

    assert observation[cell_start : cell_start + 4] == (2, 1, 0, 0)


def test_state_to_observation_encodes_current_player_hand() -> None:
    # 現在プレイヤーの手札だけが固定スロットで入ることを確認する。
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

    assert observation[hand_start : hand_start + 6] == (1, 0, 0, 1, 0, 4)
    assert observation[hand_start + 6 : hand_start + 12] == (0, 0, 0, 0, 0, 0)


def test_state_to_observation_encodes_player_phase_and_scalars() -> None:
    # プレイヤー概要、現在プレイヤー、フェーズ、スカラーが入ることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),), loss_score=5),
            PlayerState(negative_cards=(Card(Color.BLUE, 1),), loss_score=8),
            PlayerState(),
            PlayerState(),
        ),
        current_player_index=1,
        phase=Phase.REFILL,
        cards_played_this_turn=2,
        deck=(Card(Color.YELLOW, 0), Card(Color.GREEN, 1)),
        settlement_count=3,
    )

    observation = state_to_observation(state)
    player_start = BOARD_OBSERVATION_SIZE + HAND_OBSERVATION_SIZE
    current_player_start = player_start + 20
    phase_start = current_player_start + 5
    scalar_start = phase_start + 3

    assert observation[player_start : player_start + 8] == (1, 1, 0, 5, 1, 0, 1, 8)
    assert observation[current_player_start : current_player_start + 5] == (
        0,
        1,
        0,
        0,
        0,
    )
    assert observation[phase_start : phase_start + 3] == (0, 1, 0)
    assert observation[scalar_start : scalar_start + 4] == (2, 2, 3, 4)
