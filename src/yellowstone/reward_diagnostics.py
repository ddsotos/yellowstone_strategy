"""Reward diagnostics for baseline policies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

from yellowstone.action_space import action_to_index, legal_action_indices
from yellowstone.bots import HeuristicBot
from yellowstone.env import YellowstoneEnv
from yellowstone.types import Phase


@dataclass(frozen=True, slots=True)
class RewardEpisodeResult:
    """Reward and outcome for one learner episode."""

    seed: int
    total_reward: float
    step_count: int
    learner_won: bool
    learner_loss_score: int
    game_over: bool


@dataclass(frozen=True, slots=True)
class RewardDiagnostics:
    """Aggregate reward diagnostics across seeds."""

    policy: str
    episode_count: int
    average_total_reward: float
    min_total_reward: float
    max_total_reward: float
    win_rate: float
    average_step_count: float
    average_learner_loss_score: float
    episodes: tuple[RewardEpisodeResult, ...]


def run_reward_diagnostics(
    *,
    seeds: tuple[int, ...],
    policy: str = "heuristic",
    max_steps: int = 10_000,
) -> RewardDiagnostics:
    """Run baseline episodes and summarize reward behavior."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    episodes = tuple(
        _run_episode(seed=seed, policy=policy, max_steps=max_steps) for seed in seeds
    )
    episode_count = len(episodes)
    return RewardDiagnostics(
        policy=policy,
        episode_count=episode_count,
        average_total_reward=sum(episode.total_reward for episode in episodes)
        / episode_count,
        min_total_reward=min(episode.total_reward for episode in episodes),
        max_total_reward=max(episode.total_reward for episode in episodes),
        win_rate=sum(1 for episode in episodes if episode.learner_won) / episode_count,
        average_step_count=sum(episode.step_count for episode in episodes)
        / episode_count,
        average_learner_loss_score=sum(
            episode.learner_loss_score for episode in episodes
        )
        / episode_count,
        episodes=episodes,
    )


def write_reward_diagnostics(
    diagnostics: RewardDiagnostics,
    *,
    json_output: Path | None = None,
    csv_output: Path | None = None,
) -> None:
    """Write reward diagnostics as JSON and/or CSV."""
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(asdict(diagnostics), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "seed",
                    "total_reward",
                    "step_count",
                    "learner_won",
                    "learner_loss_score",
                    "game_over",
                ],
            )
            writer.writeheader()
            for episode in diagnostics.episodes:
                writer.writerow(asdict(episode))


def _run_episode(
    *,
    seed: int,
    policy: str,
    max_steps: int,
) -> RewardEpisodeResult:
    env = YellowstoneEnv()
    rng = Random(seed)
    env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    done = False
    step = None
    while not done and step_count < max_steps:
        action_index = _choose_action_index(env, policy=policy, rng=rng)
        step = env.step(action_index)
        total_reward += step.reward
        done = step.done
        step_count += 1
    if env.state is None:
        raise RuntimeError("environment did not initialize")
    learner = env.state.players[env.learning_player_index]
    return RewardEpisodeResult(
        seed=seed,
        total_reward=total_reward,
        step_count=step_count,
        learner_won=env.learning_player_index in env.state.winners,
        learner_loss_score=learner.loss_score,
        game_over=env.state.phase == Phase.GAME_OVER,
    )


def _choose_action_index(
    env: YellowstoneEnv,
    *,
    policy: str,
    rng: Random,
) -> int:
    if env.state is None:
        raise RuntimeError("environment did not initialize")
    if policy == "random":
        return rng.choice(legal_action_indices(env.state))
    if policy == "heuristic":
        action = HeuristicBot().choose_action(env.state)
        if action is None:
            raise RuntimeError("no legal action for heuristic learner")
        return action_to_index(action)
    raise ValueError("policy must be 'heuristic' or 'random'")


def _parse_args() -> tuple[tuple[int, ...], str, int, Path | None, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    return seeds, args.policy, args.max_steps, args.json_output, args.csv_output


def main() -> None:
    """Run reward diagnostics from command-line arguments."""
    seeds, policy, max_steps, json_output, csv_output = _parse_args()
    diagnostics = run_reward_diagnostics(
        seeds=seeds,
        policy=policy,
        max_steps=max_steps,
    )
    write_reward_diagnostics(
        diagnostics,
        json_output=json_output,
        csv_output=csv_output,
    )
    print(f"policy={diagnostics.policy}")
    print(f"episodes={diagnostics.episode_count}")
    print(f"average_total_reward={diagnostics.average_total_reward:.3f}")
    print(f"win_rate={diagnostics.win_rate:.3f}")


if __name__ == "__main__":
    main()
