from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from kalah_zero.agents import Agent
from kalah_zero.game import GameState


@dataclass(frozen=True, slots=True)
class GameRecord:
    winner: int | None
    final_state: GameState
    moves: int


def play_game(
    agent_0: Agent,
    agent_1: Agent,
    pits: int = 6,
    stones: int = 4,
    max_moves: int = 500,
) -> GameRecord:
    state = GameState.new_game(pits=pits, stones=stones)
    agents = [agent_0, agent_1]
    moves = 0
    while not state.is_terminal() and moves < max_moves:
        action = agents[state.current_player].select_action(state)
        state = state.apply(action)
        moves += 1
    score_0 = state.score_for_player(0)
    score_1 = state.score_for_player(1)
    winner = 0 if score_0 > score_1 else 1 if score_1 > score_0 else None
    return GameRecord(winner=winner, final_state=state, moves=moves)


@dataclass(frozen=True, slots=True)
class ArenaResult:
    games: int
    wins_0: int
    wins_1: int
    draws: int

    @property
    def win_rate_0(self) -> float:
        return self.wins_0 / max(1, self.games)


def arena(
    agent_a: Agent,
    agent_b: Agent,
    games: int = 20,
    pits: int = 6,
    stones: int = 4,
    seed: int = 0,
    on_game_complete: Callable[[int, ArenaResult], None] | None = None,
) -> ArenaResult:
    rng = random.Random(seed)
    wins_a = 0
    wins_b = 0
    draws = 0
    for index in range(games):
        if index % 2 == 0:
            record = play_game(agent_a, agent_b, pits=pits, stones=stones)
            winner_is_a = record.winner == 0
            winner_is_b = record.winner == 1
        else:
            _ = rng.random()
            record = play_game(agent_b, agent_a, pits=pits, stones=stones)
            winner_is_a = record.winner == 1
            winner_is_b = record.winner == 0
        if record.winner is None:
            draws += 1
        elif winner_is_a:
            wins_a += 1
        elif winner_is_b:
            wins_b += 1
        if on_game_complete is not None:
            on_game_complete(index + 1, ArenaResult(index + 1, wins_a, wins_b, draws))
    return ArenaResult(games=games, wins_0=wins_a, wins_1=wins_b, draws=draws)
