from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from kalah_zero.game import GameState


class Evaluator(Protocol):
    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Return `(policy, value)` from the current player's perspective."""


@dataclass(slots=True)
class UniformEvaluator:
    """A tiny evaluator useful before the neural network exists."""

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        policy = [0.0] * state.pits
        legal = state.legal_actions()
        if legal:
            prob = 1.0 / len(legal)
            for action in legal:
                policy[action] = prob
        return policy, 0.0


@dataclass(slots=True)
class RolloutEvaluator:
    """Estimate value by random playouts; useful for the UCT tutorial."""

    rollouts: int = 8
    rng: random.Random = field(default_factory=random.Random)

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        policy = [0.0] * state.pits
        legal = state.legal_actions()
        if legal:
            for action in legal:
                policy[action] = 1.0 / len(legal)
        if state.is_terminal():
            return policy, state.reward_for_player(state.current_player)
        values = [self._playout(state) for _ in range(self.rollouts)]
        return policy, sum(values) / len(values)

    def _playout(self, state: GameState) -> float:
        perspective = state.current_player
        cursor = state
        while not cursor.is_terminal():
            cursor = cursor.apply(self.rng.choice(cursor.legal_actions()))
        return cursor.reward_for_player(perspective)


@dataclass(slots=True)
class SearchNode:
    state: GameState
    prior: float
    parent: SearchNode | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(frozen=True, slots=True)
class SearchResult:
    root: SearchNode
    visits: list[int]
    policy: list[float]
    value: float

    def select_action(self, temperature: float = 1.0, rng: random.Random | None = None) -> int:
        rng = rng or random
        legal = [action for action, visits in enumerate(self.visits) if visits > 0]
        if not legal:
            raise ValueError("cannot select an action without visited legal children")
        if temperature <= 0:
            return max(legal, key=lambda action: self.visits[action])
        weights = [self.visits[action] ** (1.0 / temperature) for action in legal]
        total = sum(weights)
        if total <= 0:
            return rng.choice(legal)
        threshold = rng.random() * total
        cumulative = 0.0
        for action, weight in zip(legal, weights):
            cumulative += weight
            if cumulative >= threshold:
                return action
        return legal[-1]


@dataclass(slots=True)
class MCTS:
    simulations: int = 100
    c_puct: float = 1.5
    use_puct: bool = True
    dirichlet_alpha: float | None = None
    dirichlet_epsilon: float = 0.25
    rng: random.Random = field(default_factory=random.Random)

    def search(self, state: GameState, evaluator: Evaluator) -> SearchResult:
        root = SearchNode(state=state, prior=1.0)
        self._expand(root, evaluator)
        if self.dirichlet_alpha is not None and root.children:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            path = self._select_path(root)
            leaf = path[-1]
            if leaf.state.is_terminal():
                value = leaf.state.reward_for_player(leaf.state.current_player)
            else:
                _, value = self._expand(leaf, evaluator)
            self._backup(path, value)

        visits = [0] * state.pits
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total_visits = sum(visits)
        policy = [visit / total_visits if total_visits else 0.0 for visit in visits]
        return SearchResult(root=root, visits=visits, policy=policy, value=root.mean_value)

    def _select_path(self, root: SearchNode) -> list[SearchNode]:
        path = [root]
        node = root
        while node.expanded and not node.state.is_terminal():
            action, node = max(
                node.children.items(),
                key=lambda item: self._score_child(parent=path[-1], child=item[1]),
            )
            _ = action
            path.append(node)
        return path

    def _score_child(self, parent: SearchNode, child: SearchNode) -> float:
        q = child.mean_value
        if child.state.current_player != parent.state.current_player:
            q = -q
        if self.use_puct:
            exploration = (
                self.c_puct
                * child.prior
                * math.sqrt(parent.visit_count + 1)
                / (1 + child.visit_count)
            )
        else:
            exploration = self.c_puct * math.sqrt(
                math.log(parent.visit_count + 2) / (1 + child.visit_count)
            )
        return q + exploration

    def _expand(self, node: SearchNode, evaluator: Evaluator) -> tuple[list[float], float]:
        raw_policy, value = evaluator.evaluate(node.state)
        policy = self._masked_policy(node.state, raw_policy)
        for action in node.state.legal_actions():
            if action not in node.children:
                node.children[action] = SearchNode(
                    state=node.state.apply(action),
                    prior=policy[action],
                    parent=node,
                )
        return policy, value

    def _masked_policy(self, state: GameState, policy: list[float]) -> list[float]:
        if len(policy) != state.pits:
            raise ValueError(f"policy length {len(policy)} does not match {state.pits} pits")
        legal = state.legal_actions()
        masked = [0.0] * state.pits
        for action in legal:
            masked[action] = max(0.0, float(policy[action]))
        total = sum(masked)
        if total <= 0 and legal:
            for action in legal:
                masked[action] = 1.0 / len(legal)
            return masked
        if total > 0:
            masked = [prob / total for prob in masked]
        return masked

    def _backup(self, path: list[SearchNode], leaf_value: float) -> None:
        value_for_child = leaf_value
        child_player = path[-1].state.current_player
        for node in reversed(path):
            if node.state.current_player == child_player:
                node_value = value_for_child
            else:
                node_value = -value_for_child
            node.visit_count += 1
            node.value_sum += node_value
            value_for_child = node_value
            child_player = node.state.current_player

    def _add_root_noise(self, root: SearchNode) -> None:
        assert self.dirichlet_alpha is not None
        actions = list(root.children)
        noise = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
        total = sum(noise)
        if total <= 0:
            return
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1.0 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * sample / total
            )

    def dump_tree(self, root: SearchNode, max_depth: int = 2) -> str:
        lines: list[str] = []

        def visit(node: SearchNode, depth: int, label: str) -> None:
            indent = "  " * depth
            lines.append(
                f"{indent}{label}: N={node.visit_count} Q={node.mean_value:.3f} P={node.prior:.3f}"
            )
            if depth >= max_depth:
                return
            for action, child in sorted(node.children.items()):
                visit(child, depth + 1, f"a{action}")

        visit(root, 0, "root")
        return "\n".join(lines)

