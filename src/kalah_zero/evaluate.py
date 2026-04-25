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
    initial_state: GameState | None = None,
    state_cls=GameState,
) -> GameRecord:
    state = initial_state or state_cls.new_game(pits=pits, stones=stones)
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


def random_opening(
    plies: int,
    rng: random.Random,
    pits: int = 6,
    stones: int = 4,
    state_cls=GameState,
) -> GameState:
    state = state_cls.new_game(pits=pits, stones=stones)
    for _ in range(max(0, plies)):
        if state.is_terminal():
            break
        state = state.apply(rng.choice(state.legal_actions()))
    return state


def choose_opening_plies(
    rng: random.Random,
    opening_plies: int = 0,
    opening_plies_min: int | None = None,
    opening_plies_max: int | None = None,
) -> int:
    if opening_plies_min is None and opening_plies_max is None:
        low = high = opening_plies
    else:
        low = 0 if opening_plies_min is None else opening_plies_min
        high = low if opening_plies_max is None else opening_plies_max
    if low < 0 or high < 0:
        raise ValueError("opening plies must be non-negative")
    if high < low:
        raise ValueError("opening plies max must be greater than or equal to min")
    return rng.randint(low, high)


def choose_stones(
    rng: random.Random,
    stones: int = 4,
    stones_min: int | None = None,
    stones_max: int | None = None,
) -> int:
    if stones_min is None and stones_max is None:
        low = high = stones
    else:
        low = stones if stones_min is None else stones_min
        high = low if stones_max is None else stones_max
    if low < 0 or high < 0:
        raise ValueError("stones must be non-negative")
    if high < low:
        raise ValueError("stones max must be greater than or equal to min")
    return rng.randint(low, high)


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
    stones_min: int | None = None,
    stones_max: int | None = None,
    seed: int = 0,
    opening_plies: int = 0,
    opening_plies_min: int | None = None,
    opening_plies_max: int | None = None,
    on_game_complete: Callable[[int, ArenaResult], None] | None = None,
    state_cls=GameState,
) -> ArenaResult:
    rng = random.Random(seed)
    wins_a = 0
    wins_b = 0
    draws = 0
    paired_opening: GameState | None = None
    for index in range(games):
        if index % 2 == 0:
            sampled_stones = choose_stones(
                rng,
                stones=stones,
                stones_min=stones_min,
                stones_max=stones_max,
            )
            plies = choose_opening_plies(
                rng,
                opening_plies=opening_plies,
                opening_plies_min=opening_plies_min,
                opening_plies_max=opening_plies_max,
            )
            paired_opening = (
                random_opening(plies, rng, pits=pits, stones=sampled_stones, state_cls=state_cls)
                if plies > 0
                else None
            )
        initial_state = paired_opening
        if index % 2 == 0:
            record = play_game(
                agent_a,
                agent_b,
                pits=pits,
                stones=sampled_stones,
                initial_state=initial_state,
                state_cls=state_cls,
            )
            winner_is_a = record.winner == 0
            winner_is_b = record.winner == 1
        else:
            record = play_game(
                agent_b,
                agent_a,
                pits=pits,
                stones=sampled_stones,
                initial_state=initial_state,
                state_cls=state_cls,
            )
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
