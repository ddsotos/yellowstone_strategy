"""Developer CLI entry point."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.render import render_board, render_card, render_player_summary
from yellowstone.types import (
    Action,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
)


DEFAULT_PLAYER_COUNT = 4
DEFAULT_SEED = 1


@dataclass(frozen=True, slots=True)
class TurnLog:
    """Rendered information for one completed NPC turn."""

    turn_number: int
    player_index: int
    actions: tuple[str, ...]
    observation_after_turn: str


def main() -> None:
    """Generate and browse a 4-NPC heuristic game log."""
    run_npc_game()


def run_npc_game(
    *,
    player_count: int = DEFAULT_PLAYER_COUNT,
    seed: int = DEFAULT_SEED,
) -> None:
    """Run heuristic NPCs to completion, then browse the turn log by Enter."""
    state = create_initial_state(player_count, seed=seed)
    bot = HeuristicBot()
    rng = Random(seed)
    initial_observation = render_observation(state)
    final_state, turn_logs = create_game_log(state, bot, rng=rng)

    print("Yellowstone Strategy NPC demo")
    print(f"players={player_count} seed={seed}")
    print()
    print(f"Generated {len(turn_logs)} turns.")
    if final_state.phase == Phase.GAME_OVER:
        winners_text = ",".join(str(winner) for winner in final_state.winners)
        print(f"Game over. winners={winners_text}")
    else:
        print("Game stopped before game over because no legal action was available.")

    print()
    print("Initial state")
    print()
    print(initial_observation)

    for turn_log in turn_logs:
        input("\nPress Enter to view the next turn log...")
        print()
        print(render_turn_log(turn_log))

    print()
    print("End of log.")


def create_game_log(
    state: GameState,
    bot: BotPolicy,
    *,
    rng: Random,
) -> tuple[GameState, tuple[TurnLog, ...]]:
    """Play a full NPC game first and return logs for later browsing."""
    turn_logs: list[TurnLog] = []
    turn_number = 0

    while state.phase != Phase.GAME_OVER:
        turn_number += 1
        acting_player = state.current_player_index
        state, actions = play_current_player_turn(state, bot, rng=rng)
        if not actions:
            break
        turn_logs.append(
            TurnLog(
                turn_number=turn_number,
                player_index=acting_player,
                actions=actions,
                observation_after_turn=render_observation(state),
            )
        )

    return state, tuple(turn_logs)


def play_current_player_turn(
    state: GameState,
    bot: BotPolicy,
    *,
    rng: Random,
) -> tuple[GameState, tuple[str, ...]]:
    """Apply actions until the current player's turn is complete."""
    acting_player = state.current_player_index
    actions: list[str] = []
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        if action is None:
            break

        actions.append(render_action(state, action))
        state = apply_known_legal_action(state, action, rng=rng)
        if state.current_player_index != acting_player:
            break

    return state, tuple(actions)


def render_turn_log(turn_log: TurnLog) -> str:
    """Render one already-generated turn log."""
    lines = [f"Turn {turn_log.turn_number}: P{turn_log.player_index}"]
    lines.extend(
        f"  {index}. {action_text}"
        for index, action_text in enumerate(turn_log.actions, start=1)
    )
    lines.append("")
    lines.append(turn_log.observation_after_turn)
    return "\n".join(lines)


def render_observation(state: GameState) -> str:
    """Render the board and the next actor's hand."""
    sections = [
        (
            f"phase={state.phase.value} current=P{state.current_player_index} "
            f"played={state.cards_played_this_turn} deck={len(state.deck)} "
            f"settlements={state.settlement_count}"
        ),
        render_board(state.board),
        render_player_summary(state),
    ]
    if state.phase != Phase.GAME_OVER:
        sections.append(render_current_hand(state))
    else:
        winners = ",".join(str(winner) for winner in state.winners) or "-"
        sections.append(f"winners={winners}")
    return "\n\n".join(sections)


def render_current_hand(state: GameState) -> str:
    """Render the current player's hand with indexes."""
    player = state.players[state.current_player_index]
    if not player.hand:
        return f"P{state.current_player_index} hand: (empty)"
    cards = [
        f"{index}:{render_card(card)}"
        for index, card in enumerate(player.hand)
    ]
    return f"P{state.current_player_index} hand: {' '.join(cards)}"


def render_action(state: GameState, action: Action) -> str:
    """Render one action selected by an NPC."""
    if isinstance(action, PlaceCardAction):
        card = state.players[state.current_player_index].hand[action.hand_index]
        return (
            f"place hand[{action.hand_index}]={render_card(card)} "
            f"at ({action.position.x},{action.position.y}) "
            f"frame=({action.frame.x},{action.frame.y})"
        )
    if isinstance(action, EndTurnAction):
        return "end turn after one card"
    if isinstance(action, RefillAction):
        return f"refill source={action.source.value}"
    return repr(action)


if __name__ == "__main__":
    main()
