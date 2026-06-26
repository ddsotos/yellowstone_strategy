"""Evaluate a learned one-card/two-card advantage policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from statistics import stdev

from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.action_value_evaluation import (
    ContinuingEvaluationSummary,
    _ContinuingMatchResult,
    _run_continuing_match,
    _summarize_continuing_results,
)
from yellowstone.evaluation import EvaluationSummary, evaluate_policies, make_heuristic_policies
from yellowstone.game import apply_known_legal_action
from yellowstone.heuristic_turn_plan import (
    choose_heuristic_one_card_plan,
    choose_heuristic_two_card_plan,
)
from yellowstone.observation import state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.two_card_advantage_model import (
    predict_two_card_advantage_from_observation,
)
from yellowstone.types import Action, GameState, Phase, PlaceCardAction


@dataclass(slots=True)
class LearnedTwoCardAdvantageBot:
    """Override heuristic turn length only for large predicted advantage."""

    model_path: Path
    advantage_threshold: float = 0.0
    policy_mode: str = "unrestricted"
    confirmation_model_path: Path | None = None
    confirmation_advantage_threshold: float | None = None
    pending_actions: tuple[Action, ...] = ()
    decision_count: int = 0
    two_card_decision_count: int = 0
    heuristic_fallback_count: int = 0
    directional_override_count: int = 0

    def __post_init__(self) -> None:
        if self.advantage_threshold < 0:
            raise ValueError("advantage_threshold must not be negative")
        if (
            self.confirmation_advantage_threshold is not None
            and self.confirmation_advantage_threshold < 0
        ):
            raise ValueError("confirmation_advantage_threshold must not be negative")
        if self.policy_mode not in (
            "unrestricted",
            "one_to_two_only",
            "one_to_two_always",
            "two_to_one_only",
        ):
            raise ValueError(f"unsupported policy_mode: {self.policy_mode}")

    def choose_action(self, state: GameState) -> Action | None:
        """Use regression only for turn length; keep concrete actions heuristic."""
        if self.pending_actions:
            action = self.pending_actions[0]
            self.pending_actions = self.pending_actions[1:]
            return action
        if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
            return HeuristicBot().choose_action(state)
        one_card_plan = choose_heuristic_one_card_plan(state)
        two_card_plan = choose_heuristic_two_card_plan(state)
        if one_card_plan is None or two_card_plan is None:
            return HeuristicBot().choose_action(state)
        observation = normalize_observation(
            state_to_observation(
                state,
                heuristic_bonuses=(
                    one_card_plan.bonus_score,
                    two_card_plan.bonus_score,
                ),
                heuristic_negative_deltas=(
                    one_card_plan.negative_card_delta,
                    two_card_plan.negative_card_delta,
                ),
            )
        )
        advantage = predict_two_card_advantage_from_observation(
            observation, self.model_path
        )
        self.decision_count += 1
        if (
            self.policy_mode != "one_to_two_always"
            and abs(advantage) <= self.advantage_threshold
        ):
            self.heuristic_fallback_count += 1
            return HeuristicBot().choose_action(state)
        model_two = advantage > 0 or self.policy_mode == "one_to_two_always"
        if self.confirmation_model_path is not None:
            confirmation_threshold = (
                self.advantage_threshold
                if self.confirmation_advantage_threshold is None
                else self.confirmation_advantage_threshold
            )
            confirmation_advantage = predict_two_card_advantage_from_observation(
                observation, self.confirmation_model_path
            )
            confirmation_two = confirmation_advantage > confirmation_threshold
            model_two = model_two and confirmation_two
        if self.policy_mode != "unrestricted":
            after_first = apply_known_legal_action(
                state, one_card_plan.actions[0]
            )
            heuristic_two = isinstance(
                HeuristicBot().choose_action(after_first), PlaceCardAction
            )
            override_allowed = (
                self.policy_mode in ("one_to_two_only", "one_to_two_always")
                and not heuristic_two
                and model_two
            ) or (
                self.policy_mode == "two_to_one_only"
                and heuristic_two
                and not model_two
            )
            if not override_allowed:
                self.heuristic_fallback_count += 1
                return HeuristicBot().choose_action(state)
            self.directional_override_count += 1
        selected_plan = one_card_plan
        if model_two:
            selected_plan = two_card_plan
            self.two_card_decision_count += 1
        self.pending_actions = selected_plan.actions[1:]
        return selected_plan.actions[0]


@dataclass(frozen=True, slots=True)
class TwoCardAdvantageEvaluationResult:
    """Match result for one regression threshold."""

    model_vs_heuristic: EvaluationSummary
    heuristic_only: EvaluationSummary
    advantage_threshold: float
    decision_count: int
    two_card_decision_count: int
    heuristic_fallback_count: int
    policy_mode: str
    directional_override_count: int


@dataclass(frozen=True, slots=True)
class ContinuingTwoCardAdvantageEvaluationResult:
    """Continuing-game result for one advantage threshold."""

    model_vs_heuristic: ContinuingEvaluationSummary
    heuristic_only: ContinuingEvaluationSummary
    advantage_threshold: float
    decision_count: int
    two_card_decision_count: int
    heuristic_fallback_count: int
    policy_mode: str
    confirmation_model_path: str | None
    confirmation_advantage_threshold: float | None
    directional_override_count: int
    paired_p0_loss_share_deltas: tuple[float, ...]
    paired_p0_loss_share_delta: float
    paired_p0_loss_share_ci95_low: float
    paired_p0_loss_share_ci95_high: float


def evaluate_two_card_advantage_model(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    advantage_threshold: float,
    policy_mode: str = "unrestricted",
) -> TwoCardAdvantageEvaluationResult:
    """Evaluate regression-controlled player zero against heuristic players."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    bot = LearnedTwoCardAdvantageBot(
        model_path=model_path,
        advantage_threshold=advantage_threshold,
        policy_mode=policy_mode,
    )
    policies: tuple[BotPolicy, ...] = (
        bot,
        HeuristicBot(),
        HeuristicBot(),
        HeuristicBot(),
    )
    model_summary, _ = evaluate_policies(policies, seeds=seeds)
    heuristic_summary, _ = evaluate_policies(make_heuristic_policies(), seeds=seeds)
    return TwoCardAdvantageEvaluationResult(
        model_vs_heuristic=model_summary,
        heuristic_only=heuristic_summary,
        advantage_threshold=advantage_threshold,
        decision_count=bot.decision_count,
        two_card_decision_count=bot.two_card_decision_count,
        heuristic_fallback_count=bot.heuristic_fallback_count,
        policy_mode=policy_mode,
        directional_override_count=bot.directional_override_count,
    )


def evaluate_two_card_advantage_model_continuing(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    advantage_threshold: float,
    learner_turns: int,
    max_actions: int = 100_000,
    policy_mode: str = "unrestricted",
    confirmation_model_path: Path | None = None,
    confirmation_advantage_threshold: float | None = None,
) -> ContinuingTwoCardAdvantageEvaluationResult:
    """Evaluate the binary policy for a fixed number of continuing turns."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    if learner_turns <= 0:
        raise ValueError("learner_turns must be positive")
    model_results = []
    decision_count = 0
    two_card_decision_count = 0
    heuristic_fallback_count = 0
    directional_override_count = 0
    for seed in seeds:
        bot = LearnedTwoCardAdvantageBot(
            model_path=model_path,
            advantage_threshold=advantage_threshold,
            policy_mode=policy_mode,
            confirmation_model_path=confirmation_model_path,
            confirmation_advantage_threshold=confirmation_advantage_threshold,
        )
        model_results.append(
            _run_continuing_match(
                (bot, HeuristicBot(), HeuristicBot(), HeuristicBot()),
                seed=seed,
                learner_turns=learner_turns,
                max_actions=max_actions,
            )
        )
        decision_count += bot.decision_count
        two_card_decision_count += bot.two_card_decision_count
        heuristic_fallback_count += bot.heuristic_fallback_count
        directional_override_count += bot.directional_override_count
    heuristic_results = tuple(
        _run_continuing_match(
            (HeuristicBot(), HeuristicBot(), HeuristicBot(), HeuristicBot()),
            seed=seed,
            learner_turns=learner_turns,
            max_actions=max_actions,
        )
        for seed in seeds
    )
    (
        paired_deltas,
        paired_delta,
        paired_ci_low,
        paired_ci_high,
    ) = _paired_p0_loss_share_stats(
        tuple(model_results), heuristic_results
    )
    return ContinuingTwoCardAdvantageEvaluationResult(
        model_vs_heuristic=_summarize_continuing_results(
            tuple(model_results),
            player_count=4,
            learner_turns=learner_turns,
        ),
        heuristic_only=_summarize_continuing_results(
            heuristic_results,
            player_count=4,
            learner_turns=learner_turns,
        ),
        advantage_threshold=advantage_threshold,
        decision_count=decision_count,
        two_card_decision_count=two_card_decision_count,
        heuristic_fallback_count=heuristic_fallback_count,
        policy_mode=policy_mode,
        confirmation_model_path=(
            None
            if confirmation_model_path is None
            else str(confirmation_model_path)
        ),
        confirmation_advantage_threshold=confirmation_advantage_threshold,
        directional_override_count=directional_override_count,
        paired_p0_loss_share_deltas=paired_deltas,
        paired_p0_loss_share_delta=paired_delta,
        paired_p0_loss_share_ci95_low=paired_ci_low,
        paired_p0_loss_share_ci95_high=paired_ci_high,
    )


def evaluation_result_to_dict(
    result: TwoCardAdvantageEvaluationResult,
) -> dict[str, object]:
    """Convert an advantage evaluation to JSON-friendly data."""
    return {
        "model_vs_heuristic": asdict(result.model_vs_heuristic),
        "heuristic_only": asdict(result.heuristic_only),
        "advantage_threshold": result.advantage_threshold,
        "decision_count": result.decision_count,
        "two_card_decision_count": result.two_card_decision_count,
        "heuristic_fallback_count": result.heuristic_fallback_count,
        "policy_mode": result.policy_mode,
        "directional_override_count": result.directional_override_count,
    }


def _paired_p0_loss_share_stats(
    model_results: tuple[_ContinuingMatchResult, ...],
    heuristic_results: tuple[_ContinuingMatchResult, ...],
) -> tuple[tuple[float, ...], float, float, float]:
    deltas = tuple(
        _p0_loss_share(model_result.loss_deltas)
        - _p0_loss_share(heuristic_result.loss_deltas)
        for model_result, heuristic_result in zip(
            model_results, heuristic_results, strict=True
        )
    )
    if not deltas:
        return (), 0.0, 0.0, 0.0
    mean_delta = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return deltas, mean_delta, mean_delta, mean_delta
    margin = 1.96 * stdev(deltas) / sqrt(len(deltas))
    return deltas, mean_delta, mean_delta - margin, mean_delta + margin


def _p0_loss_share(loss_deltas: tuple[int, ...]) -> float:
    total = sum(loss_deltas)
    if total <= 0:
        return 0.0
    return loss_deltas[0] / total


def continuing_evaluation_result_to_dict(
    result: ContinuingTwoCardAdvantageEvaluationResult,
) -> dict[str, object]:
    """Convert a continuing evaluation to JSON-friendly data."""
    return {
        "model_vs_heuristic": asdict(result.model_vs_heuristic),
        "heuristic_only": asdict(result.heuristic_only),
        "advantage_threshold": result.advantage_threshold,
        "decision_count": result.decision_count,
        "two_card_decision_count": result.two_card_decision_count,
        "heuristic_fallback_count": result.heuristic_fallback_count,
        "policy_mode": result.policy_mode,
        "confirmation_model_path": result.confirmation_model_path,
        "confirmation_advantage_threshold": result.confirmation_advantage_threshold,
        "directional_override_count": result.directional_override_count,
        "paired_p0_loss_share_deltas": result.paired_p0_loss_share_deltas,
        "paired_p0_loss_share_delta": result.paired_p0_loss_share_delta,
        "paired_p0_loss_share_ci95_low": result.paired_p0_loss_share_ci95_low,
        "paired_p0_loss_share_ci95_high": result.paired_p0_loss_share_ci95_high,
    }


def main() -> None:
    """CLI entry point for advantage-policy evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=600_000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--advantage-threshold", type=float, required=True)
    parser.add_argument("--continuing-learner-turns", type=int)
    parser.add_argument("--max-actions", type=int, default=100_000)
    parser.add_argument(
        "--policy-mode",
        choices=(
            "unrestricted",
            "one_to_two_only",
            "one_to_two_always",
            "two_to_one_only",
        ),
        default="unrestricted",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--confirmation-model-path", type=Path)
    parser.add_argument("--confirmation-advantage-threshold", type=float)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    if args.continuing_learner_turns is None:
        result = evaluate_two_card_advantage_model(
            args.model_path,
            seeds=seeds,
            advantage_threshold=args.advantage_threshold,
            policy_mode=args.policy_mode,
        )
        result_dict = evaluation_result_to_dict(result)
    else:
        continuing_result = evaluate_two_card_advantage_model_continuing(
            args.model_path,
            seeds=seeds,
            advantage_threshold=args.advantage_threshold,
            learner_turns=args.continuing_learner_turns,
            max_actions=args.max_actions,
            policy_mode=args.policy_mode,
            confirmation_model_path=args.confirmation_model_path,
            confirmation_advantage_threshold=args.confirmation_advantage_threshold,
        )
        result_dict = continuing_evaluation_result_to_dict(continuing_result)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
