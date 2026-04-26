from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

from kalah_zero.game import GameState


class Evaluator(Protocol):
    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Return `(policy, value)` from the current player's perspective."""


@dataclass(slots=True)
class UniformEvaluator:
    """Assign equal probability to every legal move and return value 0.

    Useful as a placeholder before the neural network is available, or to
    test the search mechanics in isolation.
    """

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Return a uniform policy over legal actions and a neutral value of 0."""
        policy = [0.0] * state.pits
        legal = state.legal_actions()
        if legal:
            prob = 1.0 / len(legal)
            for action in legal:
                policy[action] = prob
        return policy, 0.0


@dataclass(slots=True)
class RolloutEvaluator:
    """Estimate value by averaging the outcomes of several random playouts.

    This is the classic Monte Carlo approach before neural networks: instead of
    a learned value function, play the game out randomly from the current state
    and observe who wins.
    """

    rollouts: int = 8
    rng: random.Random = field(default_factory=random.Random)

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Return a uniform policy and a value estimated from random playouts."""
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
        """Play randomly from `state` to a terminal and return the reward for the original player."""
        perspective = state.current_player
        cursor = state
        while not cursor.is_terminal():
            cursor = cursor.apply(self.rng.choice(cursor.legal_actions()))
        return cursor.reward_for_player(perspective)


@dataclass(slots=True)
class SearchNode:
    """One node in the MCTS tree, representing a single game state.

    Statistics are stored on child nodes rather than on edges. Because each
    node has exactly one parent in a tree, storing stats on the child is
    equivalent to storing them on the incoming edge.
    """

    state: GameState
    prior: float          # move probability from the evaluator, set before any visits
    parent: SearchNode | None = None  # stored for external inspection; not used during backup
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)  # keyed by local action number

    @property
    def expanded(self) -> bool:
        """Return True if this node's children have been created."""
        return bool(self.children)

    @property
    def mean_value(self) -> float:
        """Return the average backed-up value, or 0 if the node has never been visited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The output of one MCTS search: the tree root, visit counts, and derived policy."""

    root: SearchNode
    visits: list[int]    # visit_count for each action index; 0 for illegal actions
    policy: list[float]  # visits normalised to sum to 1; used as the training target
    value: float         # mean_value of the root node

    def select_action(self, temperature: float = 1.0, rng: random.Random | None = None) -> int:
        """Sample an action from the visit distribution at the given temperature.

        Temperature controls how greedy the selection is:
        - temperature <= 0: always pick the most-visited action.
        - temperature = 1: sample proportional to visit counts.
        - temperature -> 0: concentrate mass on the best action.
        """
        rng = rng or random
        legal = [action for action, visits in enumerate(self.visits) if visits > 0]
        if not legal:
            raise ValueError("cannot select an action without visited legal children")
        if temperature <= 0:
            return max(legal, key=lambda action: self.visits[action])
        # Raise visit counts to the power 1/temperature, then sample proportionally.
        weights = [self.visits[action] ** (1.0 / temperature) for action in legal]
        total = sum(weights)
        if total <= 0:
            return rng.choice(legal)
        # Manual weighted sampling via a cumulative threshold (equivalent to random.choices).
        threshold = rng.random() * total
        cumulative = 0.0
        for action, weight in zip(legal, weights):
            cumulative += weight
            if cumulative >= threshold:
                return action
        return legal[-1]  # fallback for floating-point rounding


@dataclass(slots=True)
class MCTS:
    """Monte Carlo Tree Search with UCT or PUCT selection and optional Dirichlet noise."""

    simulations: int = 100       # number of search iterations per move
    c_puct: float = 1.5          # exploration constant; larger means more exploration
    use_puct: bool = True        # True for AlphaZero-style PUCT, False for plain UCT
    dirichlet_alpha: float | None = None   # if set, add Dirichlet noise at the root during self-play
    dirichlet_epsilon: float = 0.25        # weight of noise vs. prior at the root
    rng: random.Random = field(default_factory=random.Random)

    def search(
        self,
        state: GameState,
        evaluator: Evaluator,
        callback: Callable[[SearchNode, int, list[SearchNode]], None] | None = None,
    ) -> SearchResult:
        """Run MCTS from `state` and return the visit distribution over actions.

        If `callback` is provided it is called after every simulation with the
        root node, the current simulation number (1-based), and the path of
        nodes visited during that simulation (root first, leaf last).
        """
        root = SearchNode(state=state, prior=1.0)
        self._expand(root, evaluator)
        if self.dirichlet_alpha is not None and root.children:
            self._add_root_noise(root)

        for i in range(self.simulations):
            path = self._select_path(root)
            leaf = path[-1]
            if leaf.state.is_terminal():
                value = leaf.state.reward_for_player(leaf.state.current_player)
            else:
                _, value = self._expand(leaf, evaluator)
            self._backup(path, value)
            if callback is not None:
                callback(root, i + 1, path)

        visits = [0] * state.pits
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total_visits = sum(visits)
        policy = [visit / total_visits if total_visits else 0.0 for visit in visits]
        return SearchResult(root=root, visits=visits, policy=policy, value=root.mean_value)

    def _select_path(self, root: SearchNode) -> list[SearchNode]:
        """Walk from the root to a leaf by greedily following the highest UCT/PUCT score."""
        path = [root]
        node = root
        while node.expanded and not node.state.is_terminal():
            action, node = max(
                node.children.items(),
                key=lambda item: self._score_child(parent=path[-1], child=item[1]),
            )
            _ = action  # action is selected implicitly; only the child node is needed
            path.append(node)
        return path

    def _score_child(self, parent: SearchNode, child: SearchNode) -> float:
        """Compute the UCT or PUCT score used to select which child to visit next."""
        q = child.mean_value
        # Values are stored from the perspective of the node's own current player.
        # If the child belongs to the opponent, we must negate q (zero-sum game).
        if child.state.current_player != parent.state.current_player:
            q = -q
        if self.use_puct:
            # PUCT: scale exploration by the network prior and parent visit count.
            exploration = (
                self.c_puct
                * child.prior
                * math.sqrt(parent.visit_count + 1)
                / (1 + child.visit_count)
            )
        else:
            # UCT: exploration bonus derived from the multi-armed bandit literature.
            exploration = self.c_puct * math.sqrt(
                math.log(parent.visit_count + 2) / (1 + child.visit_count)
            )
        return q + exploration

    def _expand(self, node: SearchNode, evaluator: Evaluator) -> tuple[list[float], float]:
        """Evaluate `node` and create its children, one per legal action."""
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
        """Zero out illegal actions and renormalise so legal priors sum to 1.

        Falls back to a uniform distribution if the evaluator assigns zero
        probability to every legal action (which can happen early in training).
        """
        if len(policy) != state.pits:
            raise ValueError(f"policy length {len(policy)} does not match {state.pits} pits")
        legal = state.legal_actions()
        masked = [0.0] * state.pits
        for action in legal:
            masked[action] = max(0.0, float(policy[action]))
        total = sum(masked)
        if total <= 0 and legal:
            # Evaluator gave no useful signal; fall back to uniform over legal moves.
            for action in legal:
                masked[action] = 1.0 / len(legal)
            return masked
        if total > 0:
            masked = [prob / total for prob in masked]
        return masked

    def _backup(self, path: list[SearchNode], leaf_value: float) -> None:
        """Propagate the leaf value back up the path, updating each node's statistics.

        Values are stored from the perspective of each node's current player,
        so the sign flips whenever the player changes along the path.
        """
        value_for_child = leaf_value
        child_player = path[-1].state.current_player
        for node in reversed(path):
            if node.state.current_player == child_player:
                node_value = value_for_child
            else:
                # Opponent's node: a good result for the child is bad for this node.
                node_value = -value_for_child
            node.visit_count += 1
            node.value_sum += node_value
            value_for_child = node_value
            child_player = node.state.current_player

    def _add_root_noise(self, root: SearchNode) -> None:
        """Mix Dirichlet noise into the root priors to encourage exploration in self-play.

        A Dirichlet sample is generated via the Gamma-distribution trick:
        if X_i ~ Gamma(alpha, 1) independently, then X / sum(X) ~ Dirichlet(alpha).
        """
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
        """Return a multi-line string summarising visit counts, values, and priors in the tree."""
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
