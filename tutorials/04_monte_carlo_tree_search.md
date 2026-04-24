# 04. Monte Carlo Tree Search

## Goal

This lesson introduces Monte Carlo Tree Search, the search procedure that
AlphaZero modifies rather than replaces.

Minimax tries to evaluate every branch up to a fixed depth. MCTS instead spends
more effort on moves that look promising.

## Node Statistics

For each edge \((s,a)\), MCTS stores:

\[
N(s,a) = \text{number of visits},
\]

\[
W(s,a) = \text{sum of backed-up values},
\]

\[
Q(s,a) = \frac{W(s,a)}{N(s,a)}.
\]

In code, each child node stores these quantities:

```python
@dataclass(slots=True)
class SearchNode:
    state: GameState
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

## UCT Selection

Classic UCT chooses the child with the largest score:

\[
\operatorname{UCT}(s,a) =
Q(s,a)
+ c
\sqrt{
\frac{\log(N(s)+2)}{1 + N(s,a)}
}.
\]

The exploitation term \(Q(s,a)\) prefers moves that have worked. The
exploration term is large when \(N(s,a)\) is small.

The implementation can switch between UCT and PUCT:

```python
if self.use_puct:
    exploration = self.c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (1 + child.visit_count)
else:
    exploration = self.c_puct * math.sqrt(
        math.log(parent.visit_count + 2) / (1 + child.visit_count)
    )
return q + exploration
```

## The Four Phases

One simulation consists of:

\[
\text{selection} \rightarrow \text{expansion} \rightarrow
\text{evaluation} \rightarrow \text{backup}.
\]

The code mirrors that sequence:

```python
for _ in range(self.simulations):
    path = self._select_path(root)
    leaf = path[-1]
    if leaf.state.is_terminal():
        value = leaf.state.reward_for_player(leaf.state.current_player)
    else:
        _, value = self._expand(leaf, evaluator)
    self._backup(path, value)
```

## Backup and Player Perspective

A leaf value is always from the leaf state's current player's perspective. When
the path crosses to the other player, the sign must flip:

\[
V_{1-p}(s) = -V_p(s)
\]

for win/loss rewards. Kalah extra turns mean the sign flips only when the
player-to-move changes.

```python
if node.state.current_player == child_player:
    node_value = value_for_child
else:
    node_value = -value_for_child
```

## Practice

Run:

```bash
python scripts/inspect_position.py --simulations 100 --tree-depth 2
```

Then rerun with `--simulations 10` and `--simulations 500`. Watch how the visit
distribution becomes less noisy.

