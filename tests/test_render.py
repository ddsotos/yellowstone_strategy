from yellowstone.game import create_initial_state
from yellowstone.render import render_board, render_card, render_stack, render_state
from yellowstone.types import Card, Color, GameState, PlayerState, Position


def test_render_card_uses_color_initial_and_display_rank() -> None:
    # カード表示が内部rank_indexではなく表示用rankを使うことを確認する。
    assert render_card(Card(Color.RED, 0)) == "R1"
    assert render_card(Card(Color.BLUE, 6)) == "B7"


def test_render_stack_shows_top_card_and_hidden_count() -> None:
    # 重ね置きセルで一番上のカードと下の枚数が分かることを確認する。
    stack = (Card(Color.RED, 2), Card(Color.GREEN, 2), Card(Color.BLUE, 2))

    assert render_stack(stack) == "B3+2"


def test_render_board_includes_coordinates_and_stacked_cell() -> None:
    # 盤面表示に座標見出しと重ね置きセルが含まれることを確認する。
    board = {
        Position(3, 2): (Card(Color.RED, 2), Card(Color.BLUE, 2)),
    }

    rendered = render_board(board)

    assert "y/r" in rendered
    assert "0" in rendered
    assert "2/3" in rendered
    assert "B3+1" in rendered


def test_render_state_includes_game_and_player_summary() -> None:
    # 状態表示にフェーズ、山札枚数、現在プレイヤー、プレイヤー概要が含まれることを確認する。
    state = create_initial_state(4, seed=1)

    rendered = render_state(state)

    assert "phase=play" in rendered
    assert "current=P0" in rendered
    assert f"deck={len(state.deck)}" in rendered
    assert "*P0: hand=6 negative=0 loss=5" in rendered


def test_render_state_shows_winners_when_game_over() -> None:
    # 勝者がある状態ではwinner一覧が表示されることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        winners=(1, 3),
    )

    assert "winners=1,3" in render_state(state)
