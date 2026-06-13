from dataclasses import replace

import pytest

from yellowstone.game import (
    apply_action,
    board_fits_in_some_frame,
    can_place_card_at,
    create_deck,
    create_initial_state,
    legal_actions,
)
from yellowstone.types import (
    BOARD_SIZE,
    DEFAULT_LOSS_SCORE,
    HAND_SIZE,
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


def test_create_deck_has_two_of_each_card() -> None:
    # デッキが4色x7数字x各2枚で構成されることを確認する。
    deck = create_deck()

    assert len(deck) == 56
    for color in Color:
        for rank_index in range(BOARD_SIZE):
            assert deck.count(Card(color, rank_index)) == 2


def test_create_initial_state_deals_hands_and_places_initial_card() -> None:
    # 初期化で各プレイヤーに6枚配り、初期カードを対応行の中央に置くことを確認する。
    state = create_initial_state(4, seed=1)

    assert len(state.players) == 4
    assert all(len(player.hand) == HAND_SIZE for player in state.players)
    assert all(player.loss_score == DEFAULT_LOSS_SCORE for player in state.players)
    assert len(state.board) == 1
    position, stack = next(iter(state.board.items()))
    assert position == Position(x=BOARD_SIZE // 2, y=stack[0].rank_index)
    assert len(state.deck) == 56 - 4 * HAND_SIZE - 1
    assert state.current_player_index == 0
    assert state.phase == Phase.PLAY


def test_create_initial_state_rejects_unsupported_player_count() -> None:
    # 初期実装では4〜5人以外を拒否することを確認する。
    with pytest.raises(ValueError):
        create_initial_state(3, seed=1)


def test_can_place_card_at_uses_rank_index_as_row() -> None:
    # rank_index と y が一致する横列だけに配置できることを確認する。
    card = Card(Color.RED, 2)

    assert can_place_card_at({}, card, Position(0, 2))
    assert not can_place_card_at({}, card, Position(0, 3))


def test_can_place_card_at_prevents_second_column_for_existing_color() -> None:
    # 盤面上にある色は既存列にしか置けないことを確認する。
    board = {Position(3, 2): (Card(Color.RED, 2),)}

    assert can_place_card_at(board, Card(Color.RED, 4), Position(3, 4))
    assert not can_place_card_at(board, Card(Color.RED, 4), Position(2, 4))


def test_apply_place_card_stacks_without_increasing_occupied_cells() -> None:
    # 既存カードのあるマスへの配置は重ね置きになり、占有マス数は増えないことを確認する。
    red_three = Card(Color.RED, 2)
    player = PlayerState(hand=(red_three,), loss_score=5)
    state = GameState(
        players=(player, PlayerState(), PlayerState(), PlayerState()),
        board={Position(3, 2): (Card(Color.RED, 2),)},
        deck=(),
    )

    next_state = apply_action(
        state,
        PlaceCardAction(
            hand_index=0,
            position=Position(3, 2),
            frame=Frame(2, 0),
        ),
    )

    assert next_state.board[Position(3, 2)] == (Card(Color.RED, 2), red_three)
    assert len(next_state.board) == 1
    assert next_state.players[0].loss_score == 5


def test_end_turn_advances_after_one_card() -> None:
    # 1枚配置後に1枚で終了すると次プレイヤーへ進むことを確認する。
    state = replace(create_initial_state(4, seed=2), cards_played_this_turn=1)

    next_state = apply_action(state, EndTurnAction())

    assert next_state.current_player_index == 1
    assert next_state.cards_played_this_turn == 0
    assert next_state.phase == Phase.PLAY


def test_refill_none_advances_after_two_cards() -> None:
    # 2枚配置後の補充しない選択で次プレイヤーへ進むことを確認する。
    state = replace(
        create_initial_state(4, seed=3),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_action(state, RefillAction(RefillSource.NONE))

    assert next_state.current_player_index == 1
    assert next_state.cards_played_this_turn == 0
    assert next_state.phase == Phase.PLAY


def test_refill_deck_exhaustion_settles_negative_cards() -> None:
    # 山札不足の補充で決算し、マイナスカード枚数ぶん失点することを確認する。
    current_player = PlayerState(
        hand=(Card(Color.RED, 0),),
        negative_cards=(Card(Color.BLUE, 0), Card(Color.GREEN, 0)),
        loss_score=5,
    )
    state = GameState(
        players=(
            current_player,
            PlayerState(negative_cards=(Card(Color.YELLOW, 0),), loss_score=5),
            PlayerState(loss_score=5),
            PlayerState(loss_score=5),
        ),
        deck=(Card(Color.RED, 1),),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_action(state, RefillAction(RefillSource.DECK))

    assert next_state.players[0].loss_score == 7
    assert next_state.players[1].loss_score == 6
    assert all(not player.negative_cards for player in next_state.players)
    assert next_state.current_player_index == 1
    assert next_state.settlement_count == 1
    assert board_fits_in_some_frame(next_state.board)


def test_legal_actions_include_place_actions_for_current_hand() -> None:
    # 手番プレイヤーの手札から配置アクションが生成されることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={},
        deck=(),
    )

    actions = legal_actions(state)

    assert any(isinstance(action, PlaceCardAction) for action in actions)
