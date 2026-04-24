# 04. Monte Carlo Tree Search

MCTS spends search effort unevenly. Promising moves receive more visits.

Each node stores:

```text
N(s, a) = visit count
W(s, a) = total backed-up value
Q(s, a) = W(s, a) / N(s, a)
```

Classic UCT selects children using:

```text
score = Q + c * sqrt(log(N_parent + 1) / (N_child + 1))
```

The first term exploits known good moves. The second term explores uncertain
moves.

## Four MCTS Phases

1. Selection: follow high-scoring child edges.
2. Expansion: add children for legal actions.
3. Evaluation: estimate the leaf value.
4. Backup: add the value to every node on the path.

In Kalah, backup must respect extra turns. A value changes sign only when the
player to move changes.

## Code

`MCTS(use_puct=False)` uses the UCT exploration formula. `RolloutEvaluator`
estimates leaf values by random playouts.

Run:

```bash
python scripts/inspect_position.py --simulations 100 --tree-depth 2
```

