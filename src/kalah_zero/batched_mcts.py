from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, cast

from kalah_zero.game import GameState
from kalah_zero.mcts import Evaluator, MCTS, SearchNode, SearchResult


class BatchEvaluator(Protocol):
    def evaluate_batch(self, states: list[GameState]) -> list[tuple[list[float], float]]:
        """Return one `(policy, value)` pair for each input state."""


@dataclass(slots=True)
class BatchedMCTS(MCTS):
    """A faster MCTS variant that batches neural-network leaf evaluations.

    The teaching implementation in `mcts.py` evaluates one leaf at a time. This
    class keeps the same node statistics and result shape, but gathers several
    selected leaves before calling a batch-capable evaluator such as
    `NeuralEvaluator`.
    """

    simulations: int = 100
    c_puct: float = 1.5
    use_puct: bool = True
    dirichlet_alpha: float | None = None
    dirichlet_epsilon: float = 0.25
    rng: random.Random = field(default_factory=random.Random)
    batch_size: int = 32

    def search(self, state: GameState, evaluator: Evaluator) -> SearchResult:
        root = SearchNode(state=state, prior=1.0)
        self._expand(root, evaluator)
        if self.dirichlet_alpha is not None and root.children:
            self._add_root_noise(root)

        completed = 0
        while completed < self.simulations:
            limit = min(max(1, self.batch_size), self.simulations - completed)
            pending: list[list[SearchNode]] = []
            pending_states: list[GameState] = []

            for _ in range(limit):
                path = self._select_path(root)
                self._add_virtual_visit(path)
                leaf = path[-1]
                if leaf.state.is_terminal():
                    self._remove_virtual_visit(path)
                    value = leaf.state.reward_for_player(leaf.state.current_player)
                    self._backup(path, value)
                    completed += 1
                else:
                    pending.append(path)
                    pending_states.append(leaf.state)

            for path, (raw_policy, value) in zip(
                pending,
                self._evaluate_many(evaluator, pending_states),
            ):
                self._remove_virtual_visit(path)
                self._expand_with_policy(path[-1], raw_policy)
                self._backup(path, value)
                completed += 1

        visits = [0] * state.pits
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total_visits = sum(visits)
        policy = [visit / total_visits if total_visits else 0.0 for visit in visits]
        return SearchResult(root=root, visits=visits, policy=policy, value=root.mean_value)

    def _expand_with_policy(self, node: SearchNode, raw_policy: list[float]) -> list[float]:
        policy = self._masked_policy(node.state, raw_policy)
        for action in node.state.legal_actions():
            if action not in node.children:
                node.children[action] = SearchNode(
                    state=node.state.apply(action),
                    prior=policy[action],
                    parent=node,
                )
        return policy

    def _evaluate_many(
        self,
        evaluator: Evaluator,
        states: list[GameState],
    ) -> list[tuple[list[float], float]]:
        if not states:
            return []
        batch_evaluator = getattr(evaluator, "evaluate_batch", None)
        if callable(batch_evaluator):
            return cast(BatchEvaluator, evaluator).evaluate_batch(states)
        return [evaluator.evaluate(state) for state in states]

    def _add_virtual_visit(self, path: list[SearchNode]) -> None:
        for node in path:
            node.visit_count += 1

    def _remove_virtual_visit(self, path: list[SearchNode]) -> None:
        for node in path:
            node.visit_count -= 1
