# 10. Instrumentation and Debugging

## Goal

This lesson teaches you how to look inside the search. AlphaZero bugs often
produce legal-looking games with bad learning signals, so observability matters.

The most useful quantities are:

$$
N(s,a), \qquad Q(s,a), \qquad P(s,a), \qquad \pi(a \mid s).
$$

## Tree Dumps

`MCTS.dump_tree` prints the visit count, mean value, and prior:

```python
lines.append(
    f"{indent}{label}: N={node.visit_count} "
    f"Q={node.mean_value:.3f} P={node.prior:.3f}"
)
```

Run:

```bash
python scripts/inspect_position.py --simulations 200 --tree-depth 2
```

Read the output as:

$$
N = \text{how much search trusted this branch},
$$

$$
Q = \text{how good the branch looked after backup},
$$

$$
P = \text{how much the evaluator liked it before search}.
$$

## Conservation Checks

Before terminal sweeping, Kalah preserves the number of stones:

$$
\sum_i b_i = \text{constant}.
$$

Tests protect this kind of invariant:

```python
child = state.apply(5)
self.assertEqual(child.total_stones, state.total_stones)
```

If this fails, no amount of neural-network tuning matters; the game model is
wrong.

## Legal Mask Checks

A common bug is training or searching over illegal actions. For every state:

$$
\pi(a \mid s) = 0
\qquad
\text{for all } a \notin \mathcal{A}(s).
$$

The tests check that MCTS respects this:

```python
self.assertEqual(result.policy[0], 0.0)
self.assertEqual(result.policy[1], 0.0)
self.assertGreater(result.policy[2] + result.policy[5], 0.0)
```

## Value Calibration

For terminal states, the correct value is known:

$$
v^\*(s) = z_{\operatorname{player}(s)}(s).
$$

During debugging, inspect whether the value head gives sensible signs for
obviously winning or losing positions. Early random networks will not, but
trained checkpoints should improve.

## Practice

Run:

```bash
python scripts/inspect_position.py --board "0,0,1,0,0,1,0,0,0,5,0,0,1,0" --player 0 --simulations 50
```

This is a capture position from the tests. Check whether search gives visits to
the capturing move.

