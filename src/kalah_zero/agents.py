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
    """Choose a legal move uniformly at random."""

    rng: random.Random | None = None

    def select_action(self, state: GameState) -> int:
        """Return a random legal action."""
        # Fall back to the module-level `random` when no custom rng is supplied,
        # which makes the agent easy to use without explicit seeding.
        rng = self.rng or random
        return rng.choice(state.legal_actions())


@dataclass(slots=True)
class NoisyAgent:
    """Wrap another agent and sometimes choose a random legal move instead.

    With probability `epsilon` the wrapper ignores the base agent and picks
    uniformly at random. This simulates an imperfect human-like opponent and
    prevents evaluation matches from being entirely deterministic.
    """

    base_agent: Agent
    epsilon: float = 0.1
    rng: random.Random | None = None

    def select_action(self, state: GameState) -> int:
        """Return the base agent's action, or a random action with probability epsilon."""
        rng = self.rng or random
        if rng.random() < self.epsilon:
            return rng.choice(state.legal_actions())
        return self.base_agent.select_action(state)


@dataclass(slots=True)
class GreedyAgent:
    """Choose the move with the best immediate store margin."""

    def select_action(self, state: GameState) -> int:
        """Return the action that maximises the store margin after one move."""
        player = state.current_player
        return max(
            state.legal_actions(),
            key=lambda action: state.apply(action).normalized_store_margin(player),
        )


@dataclass(slots=True)
class MinimaxAgent:
    """Search the game tree with depth-limited minimax and alpha-beta pruning."""

    depth: int = 6

    def select_action(self, state: GameState) -> int:
        """Return the action with the highest minimax value at the configured depth."""
        # `perspective` is fixed to the player who called select_action.
        # It is passed unchanged through every recursive call so all evaluations
        # are from the same player's point of view.
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
        """Recursively compute the minimax value of `state` for `perspective`.

        `alpha` is the best value the maximising side has already secured on
        this path; `beta` is the best value the minimising side has secured.
        When alpha >= beta the current branch cannot influence the root decision
        and is pruned.
        """
        if state.is_terminal():
            return state.reward_for_player(perspective)
        if depth <= 0:
            return self._evaluate(state, perspective)

        if state.current_player == perspective:
            # Maximising: current player wants the highest value.
            value = -math.inf
            for action in state.legal_actions():
                value = max(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # minimising ancestor will never choose this branch
            return value

        # Minimising: opponent wants the lowest value for `perspective`.
        value = math.inf
        for action in state.legal_actions():
            value = min(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break  # maximising ancestor already has a better option
        return value

    def _evaluate(self, state: GameState, perspective: int) -> float:
        """Heuristic score when the depth limit is reached.

        Combines the store margin (secured stones) with a discounted pit margin
        (stones still in play), normalised by the total stone count so the
        result stays in a consistent range regardless of how many stones remain.
        """
        own_pits = sum(state.pits_for(perspective))
        other_pits = sum(state.pits_for(1 - perspective))
        store_margin = state.store_for(perspective) - state.store_for(1 - perspective)
        pit_margin = own_pits - other_pits
        # Store stones count more than pit stones because they are already secured.
        return (store_margin + 0.25 * pit_margin) / max(1, state.total_stones)


@dataclass(slots=True)
class MCTSAgent:
    """Wrap an MCTS searcher and a neural evaluator behind the Agent interface."""

    # Typed as `object` to avoid a circular import with mcts.py and network.py.
    mcts: object
    evaluator: object
    temperature: float = 0.0

    def select_action(self, state: GameState) -> int:
        """Run MCTS from `state` and return the selected action."""
        result = self.mcts.search(state, self.evaluator)
        return result.select_action(temperature=self.temperature)
