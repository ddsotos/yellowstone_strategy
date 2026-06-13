"""Performance benchmarks for reinforcement-learning simulations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from yellowstone.evaluation import make_heuristic_policies, run_match


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    game_count: int
    total_turn_count: int
    total_action_count: int
    elapsed_seconds: float

    @property
    def seconds_per_game(self) -> float:
        if self.game_count == 0:
            return 0.0
        return self.elapsed_seconds / self.game_count

    @property
    def games_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.game_count / self.elapsed_seconds


def benchmark_heuristic_games(
    *,
    game_count: int,
    seed_start: int = 0,
    max_actions: int = 10_000,
) -> BenchmarkResult:
    """Run heuristic-only games without rendering and measure speed."""
    policies = make_heuristic_policies()
    total_turn_count = 0
    total_action_count = 0
    started_at = perf_counter()
    for offset in range(game_count):
        result = run_match(
            policies,
            seed=seed_start + offset,
            max_actions=max_actions,
        )
        total_turn_count += result.turn_count
        total_action_count += result.action_count
    elapsed_seconds = perf_counter() - started_at
    return BenchmarkResult(
        game_count=game_count,
        total_turn_count=total_turn_count,
        total_action_count=total_action_count,
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    """Run a small command-line benchmark."""
    result = benchmark_heuristic_games(game_count=100)
    print(f"games={result.game_count}")
    print(f"turns={result.total_turn_count}")
    print(f"actions={result.total_action_count}")
    print(f"elapsed_seconds={result.elapsed_seconds:.6f}")
    print(f"seconds_per_game={result.seconds_per_game:.6f}")
    print(f"games_per_second={result.games_per_second:.2f}")


if __name__ == "__main__":
    main()
