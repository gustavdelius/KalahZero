from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from kalah_zero.game import GameState


class Agent(Protocol):
    def select_action(self, state: GameState) -> int:
        """Choose one legal action for the player to move."""


@dataclass(slots=True)
class RandomAgent:
    rng: random.Random | None = None

    def select_action(self, state: GameState) -> int:
        rng = self.rng or random
        return rng.choice(state.legal_actions())


@dataclass(slots=True)
class GreedyAgent:
    """Choose the move with the best immediate store margin."""

    def select_action(self, state: GameState) -> int:
        player = state.current_player
        return max(
            state.legal_actions(),
            key=lambda action: state.apply(action).normalized_store_margin(player),
        )


@dataclass(slots=True)
class MinimaxAgent:
    depth: int = 6

    def select_action(self, state: GameState) -> int:
        perspective = state.current_player
        best_action = state.legal_actions()[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        for action in state.legal_actions():
            value = self._search(state.apply(action), self.depth - 1, perspective, alpha, beta)
            if value > best_value:
                best_action = action
                best_value = value
            alpha = max(alpha, best_value)
        return best_action

    def _search(
        self,
        state: GameState,
        depth: int,
        perspective: int,
        alpha: float,
        beta: float,
    ) -> float:
        if state.is_terminal():
            return state.reward_for_player(perspective)
        if depth <= 0:
            return self._evaluate(state, perspective)

        if state.current_player == perspective:
            value = -math.inf
            for action in state.legal_actions():
                value = max(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = math.inf
        for action in state.legal_actions():
            value = min(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self, state: GameState, perspective: int) -> float:
        own_pits = sum(state.pits_for(perspective))
        other_pits = sum(state.pits_for(1 - perspective))
        store_margin = state.store_for(perspective) - state.store_for(1 - perspective)
        pit_margin = own_pits - other_pits
        return (store_margin + 0.25 * pit_margin) / max(1, state.total_stones)


@dataclass(slots=True)
class MCTSAgent:
    mcts: object
    evaluator: object
    temperature: float = 0.0

    def select_action(self, state: GameState) -> int:
        result = self.mcts.search(state, self.evaluator)
        return result.select_action(temperature=self.temperature)

