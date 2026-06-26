"""Evaluate the learned binary decision while keeping actions heuristic."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.evaluation import (
    EvaluationSummary,
    evaluate_policies,
    make_heuristic_policies,
)
from yellowstone.heuristic_turn_plan import (
    choose_heuristic_one_card_plan,
    choose_heuristic_two_card_plan,
)
from yellowstone.observation import state_to_observation
from yellowstone.observation_normalization import normalize_observation
from yellowstone.two_card_decision_model import (
    predict_two_card_probability_from_observation,
)
from yellowstone.types import Action, GameState, Phase


@dataclass(slots=True)
class LearnedTwoCardDecisionBot:
    """Use a learned binary choice and heuristic concrete turn plans."""

    model_path: Path
    threshold: float = 0.5
    confidence_margin: float = 0.0
    pending_actions: tuple[Action, ...] = ()
    decision_count: int = 0
    two_card_decision_count: int = 0
    heuristic_fallback_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= self.confidence_margin <= 0.5:
            raise ValueError("confidence_margin must be between 0 and 0.5")

    def choose_action(self, state: GameState) -> Action | None:
        """Choose only turn length with the model; delegate actions to heuristics."""
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
        probability = predict_two_card_probability_from_observation(
            observation, self.model_path
        )
        lower_threshold = self.threshold - self.confidence_margin
        upper_threshold = self.threshold + self.confidence_margin
        self.decision_count += 1
        if lower_threshold < probability < upper_threshold:
            self.heuristic_fallback_count += 1
            return HeuristicBot().choose_action(state)
        selected_plan = one_card_plan
        if probability >= upper_threshold:
            selected_plan = two_card_plan
            self.two_card_decision_count += 1
        actions = selected_plan.actions
        self.pending_actions = actions[1:]
        return actions[0]


@dataclass(frozen=True, slots=True)
class TwoCardDecisionEvaluationResult:
    """Match quality and binary decision rate for one threshold."""

    model_vs_heuristic: EvaluationSummary
    heuristic_only: EvaluationSummary
    threshold: float
    decision_count: int
    two_card_decision_count: int
    heuristic_fallback_count: int = 0
    confidence_margin: float = 0.0

    @property
    def two_card_decision_rate(self) -> float:
        if self.decision_count == 0:
            return 0.0
        return self.two_card_decision_count / self.decision_count


def evaluate_two_card_decision_model(
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    threshold: float = 0.5,
    confidence_margin: float = 0.0,
) -> TwoCardDecisionEvaluationResult:
    """Evaluate learned player zero against three heuristic players."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    learned_bot = LearnedTwoCardDecisionBot(
        model_path=model_path,
        threshold=threshold,
        confidence_margin=confidence_margin,
    )
    policies: tuple[BotPolicy, ...] = (
        learned_bot,
        HeuristicBot(),
        HeuristicBot(),
        HeuristicBot(),
    )
    model_summary, _ = evaluate_policies(policies, seeds=seeds)
    heuristic_summary, _ = evaluate_policies(make_heuristic_policies(), seeds=seeds)
    return TwoCardDecisionEvaluationResult(
        model_vs_heuristic=model_summary,
        heuristic_only=heuristic_summary,
        threshold=threshold,
        decision_count=learned_bot.decision_count,
        two_card_decision_count=learned_bot.two_card_decision_count,
        heuristic_fallback_count=learned_bot.heuristic_fallback_count,
        confidence_margin=confidence_margin,
    )


def evaluation_result_to_dict(
    result: TwoCardDecisionEvaluationResult,
) -> dict[str, object]:
    """Convert an evaluation result to JSON-friendly data."""
    return {
        "model_vs_heuristic": asdict(result.model_vs_heuristic),
        "heuristic_only": asdict(result.heuristic_only),
        "threshold": result.threshold,
        "confidence_margin": result.confidence_margin,
        "decision_count": result.decision_count,
        "two_card_decision_count": result.two_card_decision_count,
        "heuristic_fallback_count": result.heuristic_fallback_count,
        "two_card_decision_rate": result.two_card_decision_rate,
    }


def main() -> None:
    """CLI entry point for learned binary decision evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    result = evaluate_two_card_decision_model(
        args.model_path,
        seeds=seeds,
        threshold=args.threshold,
        confidence_margin=args.confidence_margin,
    )
    result_dict = evaluation_result_to_dict(result)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
