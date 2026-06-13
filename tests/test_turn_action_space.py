from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.turn_action_space import (
    TURN_ACTION_SPACE_SIZE,
    TurnAction,
    legal_turn_action_indices,
    legal_turn_action_mask,
    resolve_turn_action,
    turn_action_from_index,
    turn_action_space_metadata,
    turn_action_to_index,
)
from yellowstone.types import EndTurnAction, PlaceCardAction, RefillAction


def test_turn_action_space_has_36_actions() -> None:
    # 1枚プレイ6通りと2枚プレイ6*5通りで、turn-level行動が36通りになることを確認する。
    metadata = turn_action_space_metadata()

    assert TURN_ACTION_SPACE_SIZE == 36
    assert metadata["one_card_action_count"] == 6
    assert metadata["two_card_action_count"] == 30


def test_turn_action_round_trips_through_index() -> None:
    # turn-level行動をindex化し、1枚/2枚の手札スロット選択を復元できることを確認する。
    assert turn_action_from_index(0) == TurnAction((0,))
    assert turn_action_to_index(TurnAction((0,))) == 0
    assert turn_action_from_index(6) == TurnAction((0, 1))
    assert turn_action_to_index(TurnAction((0, 1))) == 6
    assert turn_action_from_index(35) == TurnAction((5, 4))
    assert turn_action_to_index(TurnAction((5, 4))) == 35


def test_resolve_one_card_turn_uses_heuristic_placement_then_end_turn() -> None:
    # 1枚プレイ行動は、選んだ手札をheuristic配置してから1枚終了に変換されることを確認する。
    state = create_initial_state(4, seed=1)

    actions = resolve_turn_action(state, 0)

    assert isinstance(actions[0], PlaceCardAction)
    assert actions[0].hand_index == 0
    assert any(isinstance(action, EndTurnAction) for action in actions)


def test_resolve_two_card_turn_uses_two_selected_original_slots() -> None:
    # 2枚プレイ行動は、元の手札スロット順に2枚の配置へ変換されることを確認する。
    state = create_initial_state(4, seed=1)

    actions = resolve_turn_action(state, turn_action_to_index(TurnAction((0, 1))))
    state_after_first = apply_known_legal_action(state, actions[0])

    assert isinstance(actions[0], PlaceCardAction)
    assert isinstance(actions[1], PlaceCardAction)
    assert actions[0].hand_index == 0
    assert actions[1].hand_index == 0
    assert actions[1] in legal_actions(state_after_first)
    assert any(isinstance(action, RefillAction) for action in actions)


def test_legal_turn_action_mask_matches_indices() -> None:
    # turn-level合法手maskのTrue位置が合法index一覧と一致することを確認する。
    state = create_initial_state(4, seed=1)
    indexes = set(legal_turn_action_indices(state))
    mask = legal_turn_action_mask(state)

    assert len(mask) == TURN_ACTION_SPACE_SIZE
    assert {index for index, value in enumerate(mask) if value} == indexes
