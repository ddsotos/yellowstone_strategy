from dataclasses import replace

import yellowstone.two_card_decision as two_card_decision_module
from yellowstone.two_card_decision import (
    TwoCardDecision,
    TwoCardDecisionCase,
    TwoCardDecisionValue,
    choose_two_card_decision_by_rollout,
    classify_two_card_decision_point,
    resolve_two_card_decision,
    resolve_two_card_decision_before_refill,
)
from yellowstone.bots import placement_sort_key
from yellowstone.game import apply_known_legal_action, legal_actions
from yellowstone.heuristic_turn_plan import choose_heuristic_two_card_plan
from yellowstone.types import (
    Card,
    Color,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    PlayerState,
    Position,
)


def test_classifies_heuristic_stop_with_second_available() -> None:
    # 手番開始前に、heuristicは1枚で止めるが2枚プレイも可能な局面を識別する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0), Card(Color.BLUE, 6))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 0): (Card(Color.RED, 0),),
            Position(0, 1): (Card(Color.RED, 1),),
            Position(0, 2): (Card(Color.RED, 2),),
        },
        phase=Phase.PLAY,
        cards_played_this_turn=0,
    )

    assert (
        classify_two_card_decision_point(state)
        == TwoCardDecisionCase.HEURISTIC_STOPS_WITH_SECOND_AVAILABLE
    )
    assert isinstance(
        resolve_two_card_decision_before_refill(
            state,
            TwoCardDecision.STOP_AFTER_ONE,
        )[0],
        PlaceCardAction,
    )
    assert isinstance(
        resolve_two_card_decision_before_refill(
            state,
            TwoCardDecision.STOP_AFTER_ONE,
        )[1],
        EndTurnAction,
    )
    assert len(resolve_two_card_decision_before_refill(
        state,
        TwoCardDecision.PLAY_SECOND_HEURISTIC,
    )) == 2


def test_classifies_heuristic_damaging_second_card() -> None:
    # 手番開始前に、heuristicが失点カードを増やしてでも2枚出す局面を識別する。
    state = _damaging_positive_second_state()

    assert (
        classify_two_card_decision_point(state)
        == TwoCardDecisionCase.HEURISTIC_PLAYS_DAMAGING_SECOND
    )
    plan = choose_heuristic_two_card_plan(state)
    assert plan is not None
    assert plan.negative_card_delta > 0


def test_rollout_oracle_scores_damaging_two_card_turn() -> None:
    # 4手番先を見る設定を持ち、失点あり2枚プレイも比較対象に入れられる。
    state = _damaging_positive_second_state()

    result = choose_two_card_decision_by_rollout(
        state,
        horizon_learner_turns=0,
        target="self_loss",
    )

    values = {value.decision: value.target_self_loss for value in result.values}
    assert values[TwoCardDecision.PLAY_SECOND_HEURISTIC] <= values[
        TwoCardDecision.STOP_AFTER_ONE
    ]


def test_rollout_oracle_averages_common_random_seeds(monkeypatch) -> None:
    # 1枚案と2枚案を同じ複数seedで評価し、その平均を教師値にする。
    state = _damaging_positive_second_state()
    calls: list[tuple[TwoCardDecision, int]] = []

    def fake_evaluate_decision(*args, **kwargs):
        decision = kwargs["decision"]
        seed = kwargs["seed"]
        calls.append((decision, seed))
        decision_offset = (
            0.0 if decision == TwoCardDecision.STOP_AFTER_ONE else 10.0
        )
        return TwoCardDecisionValue(
            decision=decision,
            target_self_loss=float(seed),
            target_relative_loss=float(seed) + decision_offset,
        )

    monkeypatch.setattr(
        two_card_decision_module,
        "_evaluate_decision",
        fake_evaluate_decision,
    )

    result = choose_two_card_decision_by_rollout(
        state,
        seed=1,
        rollout_count=3,
    )

    assert {value.target_self_loss for value in result.values} == {1_000_004.0}
    assert result.selected_decision == TwoCardDecision.STOP_AFTER_ONE
    assert calls == [
        (TwoCardDecision.STOP_AFTER_ONE, 1),
        (TwoCardDecision.STOP_AFTER_ONE, 1_000_004),
        (TwoCardDecision.STOP_AFTER_ONE, 2_000_007),
        (TwoCardDecision.PLAY_SECOND_HEURISTIC, 1),
        (TwoCardDecision.PLAY_SECOND_HEURISTIC, 1_000_004),
        (TwoCardDecision.PLAY_SECOND_HEURISTIC, 2_000_007),
    ]


def test_resolve_two_card_decision_appends_refill_for_two_card_turn() -> None:
    # 2枚プレイ判断は2枚配置後にheuristic補充まで解決できる。
    state = _damaging_positive_second_state()

    actions = resolve_two_card_decision(state, TwoCardDecision.PLAY_SECOND_HEURISTIC)

    assert len(actions) == 3


def test_two_card_plan_matches_exhaustive_state_transition_search() -> None:
    # 高速化後も全候補で状態遷移する従来方式と同じ2枚を選ぶ。
    state = _damaging_positive_second_state()

    plan = choose_heuristic_two_card_plan(state)
    expected_actions, expected_score, expected_negative = _exhaustive_two_card_plan(
        state
    )

    assert plan is not None
    assert plan.actions == expected_actions
    assert plan.bonus_score == expected_score
    assert plan.negative_card_delta == expected_negative


def test_two_card_plan_caps_bonus_at_remaining_loss_score() -> None:
    # 失点が少ない局面でも解析計算と実際の状態遷移のボーナスを一致させる。
    state = _damaging_positive_second_state()
    player = replace(state.players[0], loss_score=1)
    state = replace(state, players=(player, *state.players[1:]))

    plan = choose_heuristic_two_card_plan(state)
    expected_actions, expected_score, expected_negative = _exhaustive_two_card_plan(
        state
    )

    assert plan is not None
    assert plan.actions == expected_actions
    assert plan.bonus_score == expected_score
    assert plan.negative_card_delta == expected_negative


def _damaging_positive_second_state() -> GameState:
    return GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0), Card(Color.BLUE, 2)), loss_score=5),
            PlayerState(loss_score=5),
            PlayerState(loss_score=5),
            PlayerState(loss_score=5),
        ),
        board={
            Position(0, 0): (Card(Color.RED, 0),),
            Position(1, 0): (Card(Color.GREEN, 0),),
            Position(2, 0): (Card(Color.BLUE, 0),),
            Position(0, 1): (Card(Color.RED, 1),),
            Position(1, 1): (Card(Color.GREEN, 1),),
            Position(2, 1): (Card(Color.BLUE, 1),),
            Position(0, 2): (Card(Color.RED, 2),),
            Position(1, 2): (Card(Color.GREEN, 2),),
            Position(6, 6): (Card(Color.YELLOW, 6),),
        },
        phase=Phase.PLAY,
        cards_played_this_turn=0,
    )


def _exhaustive_two_card_plan(
    state: GameState,
) -> tuple[tuple[PlaceCardAction, PlaceCardAction], int, int]:
    candidates = []
    player_index = state.current_player_index
    for first_action in _place_actions(state):
        after_first = apply_known_legal_action(state, first_action)
        for second_action in _place_actions(after_first):
            after_second = apply_known_legal_action(after_first, second_action)
            negative_delta = (
                len(after_second.players[player_index].negative_cards)
                - len(state.players[player_index].negative_cards)
            )
            score_gain = (
                state.players[player_index].loss_score
                - after_second.players[player_index].loss_score
            )
            damage_priority = 0 if negative_delta == 0 else 1
            candidates.append(
                (
                    (
                        damage_priority,
                        negative_delta if damage_priority == 0 else -score_gain,
                        0 if damage_priority == 0 else negative_delta,
                        *placement_sort_key(state, first_action),
                        *placement_sort_key(after_first, second_action),
                    ),
                    (first_action, second_action),
                    score_gain,
                    negative_delta,
                )
            )
    _, actions, score_gain, negative_delta = min(
        candidates, key=lambda candidate: candidate[0]
    )
    return actions, score_gain, negative_delta


def _place_actions(state: GameState) -> tuple[PlaceCardAction, ...]:
    return tuple(
        action for action in legal_actions(state) if isinstance(action, PlaceCardAction)
    )
