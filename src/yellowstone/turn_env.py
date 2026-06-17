"""Turn-level reinforcement-learning environment."""

from __future__ import annotations

from dataclasses import dataclass

from yellowstone.env import EnvReset, EnvStep, YellowstoneEnv
from yellowstone.game import apply_known_legal_action
from yellowstone.observation import state_to_observation
from yellowstone.rewards import (
    learned_state_value_reward,
    reward_for_transition,
    turn_action_reward,
)
from yellowstone.turn_action_space import (
    legal_turn_action_mask,
    resolve_turn_action,
)
from yellowstone.types import Phase


@dataclass(slots=True)
class YellowstoneTurnEnv(YellowstoneEnv):
    """Environment where the learner chooses one-card or two-card turn plans."""

    def reset(self, seed: int | None = None) -> EnvReset:
        """Start a new game and return a turn-level action mask."""
        result = YellowstoneEnv.reset(self, seed=seed)
        if self.state is None:
            raise RuntimeError("reset did not initialize state")
        return EnvReset(
            observation=result.observation,
            legal_action_mask=legal_turn_action_mask(self.state),
            info=self._info(),
        )

    def step(self, action_index: int) -> EnvStep:
        """Apply one learner turn plan, then auto-advance NPC turns."""
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self.state.phase == Phase.GAME_OVER:
            raise RuntimeError("cannot step after game over")
        if self.state.current_player_index != self.learning_player_index:
            raise RuntimeError("environment is not waiting for the learner")

        before = self.state
        for action in resolve_turn_action(self.state, action_index):
            self.state = apply_known_legal_action(self.state, action, rng=self.rng)
            if self.state.phase == Phase.GAME_OVER:
                break
        after_learner_turn = self.state
        if self.state.phase != Phase.GAME_OVER:
            self.state = self._advance_npcs(self.state)

        done = self.state.phase == Phase.GAME_OVER or self.stopped_reason is not None
        reward = reward_for_transition(
            before,
            self.state,
            player_index=self.learning_player_index,
            include_learned_state_value=False,
        )
        reward += learned_state_value_reward(
            before,
            after_learner_turn,
            player_index=self.learning_player_index,
            allow_after_player_perspective=True,
        )
        reward += turn_action_reward(
            before,
            after_learner_turn,
            action_index=action_index,
            player_index=self.learning_player_index,
        )
        return EnvStep(
            observation=state_to_observation(self.state),
            reward=reward,
            done=done,
            legal_action_mask=legal_turn_action_mask(self.state),
            info=self._info(),
        )
