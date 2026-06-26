"""Binary turn-start decisions for playing one card or two cards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random

from yellowstone.action_value_dataset import _rollout_horizon
from yellowstone.bots import HeuristicBot
from yellowstone.heuristic_turn_plan import (
    HeuristicTurnPlan,
    choose_heuristic_one_card_plan,
    choose_heuristic_two_card_plan,
    negative_card_delta_after_actions,
)
from yellowstone.game import apply_known_legal_action
from yellowstone.types import (
    Action,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
)


class TwoCardDecision(Enum):
    """Whether to play a one-card or two-card heuristic turn."""

    STOP_AFTER_ONE = "stop_after_one"
    PLAY_SECOND_HEURISTIC = "play_second_heuristic"


class TwoCardDecisionCase(Enum):
    """Known second-card decision patterns."""

    NOT_APPLICABLE = "not_applicable"
    HEURISTIC_STOPS_WITH_SECOND_AVAILABLE = "heuristic_stops_with_second_available"
    HEURISTIC_PLAYS_NO_DAMAGE_SECOND = "heuristic_plays_no_damage_second"
    HEURISTIC_PLAYS_DAMAGING_SECOND = "heuristic_plays_damaging_second"


@dataclass(frozen=True, slots=True)
class TwoCardDecisionValue:
    """Finite-horizon value for one binary turn-start decision."""

    decision: TwoCardDecision
    target_self_loss: float
    target_relative_loss: float


@dataclass(frozen=True, slots=True)
class TwoCardDecisionResult:
    """Rollout comparison result for the two turn-start decisions."""

    selected_decision: TwoCardDecision
    values: tuple[TwoCardDecisionValue, ...]


def classify_two_card_decision_point(state: GameState) -> TwoCardDecisionCase:
    """Classify a turn-start state for the binary one-card/two-card decision."""
    if not _is_decision_point(state):
        return TwoCardDecisionCase.NOT_APPLICABLE
    two_card_plan = choose_heuristic_two_card_plan(state)
    return classify_two_card_decision_point_with_plan(state, two_card_plan)


def classify_two_card_decision_point_with_plan(
    state: GameState,
    two_card_plan: HeuristicTurnPlan | None,
) -> TwoCardDecisionCase:
    """Classify a decision point using an already computed two-card plan."""
    if not _is_decision_point(state):
        return TwoCardDecisionCase.NOT_APPLICABLE
    if two_card_plan is None:
        return TwoCardDecisionCase.NOT_APPLICABLE

    heuristic_actions = _resolve_standard_heuristic_turn_before_refill(state)
    heuristic_place_count = sum(
        1 for action in heuristic_actions if isinstance(action, PlaceCardAction)
    )
    if heuristic_place_count == 1:
        return TwoCardDecisionCase.HEURISTIC_STOPS_WITH_SECOND_AVAILABLE
    if heuristic_place_count != 2:
        return TwoCardDecisionCase.NOT_APPLICABLE
    if negative_card_delta_after_actions(state, two_card_plan.actions) > 0:
        return TwoCardDecisionCase.HEURISTIC_PLAYS_DAMAGING_SECOND
    return TwoCardDecisionCase.HEURISTIC_PLAYS_NO_DAMAGE_SECOND


def choose_heuristic_one_card_action(state: GameState) -> PlaceCardAction | None:
    """Choose the first card of a one-card turn using the standard heuristic."""
    plan = choose_heuristic_one_card_plan(state)
    if plan is None or not isinstance(plan.actions[0], PlaceCardAction):
        return None
    return plan.actions[0]


def choose_heuristic_two_card_actions(
    state: GameState,
) -> tuple[PlaceCardAction, PlaceCardAction] | None:
    """Choose a two-card turn from the turn-start state with heuristic scoring."""
    plan = choose_heuristic_two_card_plan(state)
    if plan is None:
        return None
    first_action, second_action = plan.actions
    if not isinstance(first_action, PlaceCardAction) or not isinstance(
        second_action,
        PlaceCardAction,
    ):
        return None
    return first_action, second_action


def resolve_two_card_decision_before_refill(
    state: GameState,
    decision: TwoCardDecision,
) -> tuple[Action, ...]:
    """Resolve a turn-start binary decision, stopping before any refill choice."""
    if not _is_decision_point(state):
        raise ValueError("two-card decision requires a turn-start play state")
    if decision == TwoCardDecision.STOP_AFTER_ONE:
        one_card_plan = choose_heuristic_one_card_plan(state)
        if one_card_plan is None:
            raise ValueError("one-card heuristic action is not legal")
        return one_card_plan.actions

    two_card_plan = choose_heuristic_two_card_plan(state)
    if two_card_plan is None:
        raise ValueError("playing two cards is not legal")
    return two_card_plan.actions


def resolve_two_card_decision(
    state: GameState,
    decision: TwoCardDecision,
) -> tuple[Action, ...]:
    """Resolve a binary decision and append heuristic refill when needed."""
    resolved = list(resolve_two_card_decision_before_refill(state, decision))
    next_state = state
    for action in resolved:
        next_state = apply_known_legal_action(next_state, action)
    if next_state.phase == Phase.REFILL:
        refill_action = HeuristicBot().choose_action(next_state)
        if refill_action is None:
            raise ValueError("no refill action is available")
        resolved.append(refill_action)
    return tuple(resolved)


def choose_two_card_decision_by_rollout(
    state: GameState,
    *,
    learning_player_index: int | None = None,
    horizon_learner_turns: int = 4,
    seed: int = 0,
    max_actions: int = 10_000,
    continuing_game: bool = False,
    target: str = "relative_loss",
    plans: tuple[HeuristicTurnPlan, HeuristicTurnPlan] | None = None,
    rollout_count: int = 1,
) -> TwoCardDecisionResult:
    """Choose one-card/two-card by comparing finite-horizon heuristic rollouts.

    This is an oracle-style method for dataset generation and diagnostics. The
    later learned policy should imitate this binary choice, while card and frame
    selection remain heuristic.
    """
    if not _is_decision_point(state):
        raise ValueError("two-card rollout decision requires a turn-start state")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if learning_player_index is None:
        learning_player_index = state.current_player_index
    if plans is None:
        one_card_plan = choose_heuristic_one_card_plan(state)
        two_card_plan = choose_heuristic_two_card_plan(state)
        if one_card_plan is None or two_card_plan is None:
            raise ValueError("both one-card and two-card plans are required")
        plans = (one_card_plan, two_card_plan)
    baseline_losses = tuple(player.loss_score for player in state.players)
    values = tuple(
        _average_decision_rollouts(
            state,
            decision=decision,
            baseline_losses=baseline_losses,
            learning_player_index=learning_player_index,
            horizon_learner_turns=horizon_learner_turns,
            seed=seed,
            rollout_count=rollout_count,
            max_actions=max_actions,
            continuing_game=continuing_game,
            plans=plans,
        )
        for decision in TwoCardDecision
    )
    if _target_index(target) == 0:
        selected = min(
            values,
            key=lambda value: (value.target_self_loss, value.target_relative_loss),
        )
    else:
        selected = min(
            values,
            key=lambda value: (value.target_relative_loss, value.target_self_loss),
        )
    return TwoCardDecisionResult(
        selected_decision=selected.decision,
        values=values,
    )


def _average_decision_rollouts(
    state: GameState,
    *,
    decision: TwoCardDecision,
    baseline_losses: tuple[int, ...],
    learning_player_index: int,
    horizon_learner_turns: int,
    seed: int,
    rollout_count: int,
    max_actions: int,
    continuing_game: bool,
    plans: tuple[HeuristicTurnPlan, HeuristicTurnPlan],
) -> TwoCardDecisionValue:
    values = tuple(
        _evaluate_decision(
            state,
            decision=decision,
            baseline_losses=baseline_losses,
            learning_player_index=learning_player_index,
            horizon_learner_turns=horizon_learner_turns,
            seed=seed + rollout_index * 1_000_003,
            max_actions=max_actions,
            continuing_game=continuing_game,
            plans=plans,
        )
        for rollout_index in range(rollout_count)
    )
    return TwoCardDecisionValue(
        decision=decision,
        target_self_loss=sum(value.target_self_loss for value in values)
        / rollout_count,
        target_relative_loss=sum(value.target_relative_loss for value in values)
        / rollout_count,
    )


def _evaluate_decision(
    state: GameState,
    *,
    decision: TwoCardDecision,
    baseline_losses: tuple[int, ...],
    learning_player_index: int,
    horizon_learner_turns: int,
    seed: int,
    max_actions: int,
    continuing_game: bool,
    plans: tuple[HeuristicTurnPlan, HeuristicTurnPlan],
) -> TwoCardDecisionValue:
    rng = Random(seed)
    next_state = state
    plan = plans[0] if decision == TwoCardDecision.STOP_AFTER_ONE else plans[1]
    for action in plan.actions:
        next_state = apply_known_legal_action(next_state, action, rng=rng)
    if horizon_learner_turns <= 0:
        targets = _loss_targets(
            next_state,
            baseline_losses=baseline_losses,
            learning_player_index=learning_player_index,
        )
    else:
        rollout = _rollout_horizon(
            next_state,
            baseline_losses=baseline_losses,
            learning_player_index=learning_player_index,
            horizon_learner_turns=horizon_learner_turns,
            rng=rng,
            max_actions=max_actions,
            continuing_game=continuing_game,
        )
        if rollout is None:
            raise ValueError("rollout did not reach the requested horizon")
        targets = rollout
    return TwoCardDecisionValue(
        decision=decision,
        target_self_loss=targets[0],
        target_relative_loss=targets[1],
    )


def _loss_targets(
    state: GameState,
    *,
    baseline_losses: tuple[int, ...],
    learning_player_index: int,
) -> tuple[float, float]:
    loss_deltas = tuple(
        player.loss_score - baseline_loss
        for player, baseline_loss in zip(state.players, baseline_losses, strict=True)
    )
    self_loss = float(loss_deltas[learning_player_index])
    average_loss = sum(loss_deltas) / len(loss_deltas)
    return self_loss, self_loss - average_loss


def _is_decision_point(state: GameState) -> bool:
    return state.phase == Phase.PLAY and state.cards_played_this_turn == 0


def _resolve_standard_heuristic_turn_before_refill(
    state: GameState,
) -> tuple[Action, ...]:
    next_state = state
    actions: list[Action] = []
    player_index = state.current_player_index
    while (
        next_state.phase == Phase.PLAY
        and next_state.current_player_index == player_index
    ):
        action = choose_heuristic_one_card_action(next_state)
        if next_state.cards_played_this_turn == 1:
            action = HeuristicBot().choose_action(next_state)
        if action is None:
            break
        actions.append(action)
        next_state = apply_known_legal_action(next_state, action)
        if isinstance(action, EndTurnAction):
            break
    return tuple(actions)


def _target_index(target: str) -> int:
    if target == "self_loss":
        return 0
    if target == "relative_loss":
        return 1
    raise ValueError("target must be 'self_loss' or 'relative_loss'")
