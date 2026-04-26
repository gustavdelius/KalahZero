# 10. Instrumentation and Debugging

## Goal

This lesson teaches you how to look inside the search. AlphaZero bugs often
produce legal-looking games with bad learning signals, so observability matters.
Instrumentation means adding outputs and summaries that help you inspect what
the program is doing.

The most useful quantities are:

$$
N(s,a), \qquad Q(s,a), \qquad P(s,a), \qquad \pi(a \mid s).
$$

## Tree Dumps

`MCTS.dump_tree` prints the visit count, mean value, and prior. A prior is the
move probability supplied before search has gathered much evidence:

```python
lines.append(
    f"{indent}{label}: N={node.visit_count} "
    f"Q={node.mean_value:.3f} P={node.prior:.3f}"
)
```

Without `--board` the script starts from the standard opening position (6 pits,
4 stones each). Pass a comma-separated board including both stores to inspect
any other position, as shown in the Practice section below.

All three commands below accept `--checkpoint` to use a trained network instead
of the uniform evaluator. With a trained network the priors P reflect what the
network has learned, making the output much more informative:

```bash
python scripts/inspect_position.py --checkpoint checkpoints/residual_depth.pt --simulations 200 --tree-depth 2
```

Without `--checkpoint` the uniform evaluator assigns equal probability to every
legal move, so all priors are identical and Q values converge slowly.

To watch N, Q, and P evolve simulation by simulation, add `--watch`:

```bash
python scripts/inspect_position.py --simulations 200 --tree-depth 2 --watch
```

The terminal redraws after every simulation. Use `--watch-every N` to redraw
less often (useful for large simulation counts) and `--delay SECS` to control
the pace.

To see instead which path each simulation followed, add `--trace`:

```bash
python scripts/inspect_position.py --checkpoint checkpoints/residual_depth.pt --simulations 50 --trace
```

This prints one line per simulation showing the sequence of actions from root
to the selected leaf and that leaf's prior — letting you observe how early
simulations stay shallow (one action) while later ones push deeper as the tree
fills in.

Under the hood, `MCTS.search` accepts an optional `callback(root, sim, path)`
that is called after each backup; both modes pass a closure that reads from
`path`.

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

The tests check that Monte Carlo Tree Search (MCTS) respects this:

```python
self.assertEqual(result.policy[0], 0.0)
self.assertEqual(result.policy[1], 0.0)
self.assertGreater(result.policy[2] + result.policy[5], 0.0)
```

## Value Calibration

For terminal states, the correct value is known:

$$
v^*(s) = z_{\operatorname{player}(s)}(s).
$$

During debugging, inspect whether the value head gives sensible signs for
obviously winning or losing positions. Early random networks will not, but
trained checkpoints should improve.

## Practice

The `--board` string is the flat board tuple in index order:

```
P0 pits (0‥5),  P0 store (6),  P1 pits (7‥12),  P1 store (13)
```

Run:

```bash
python scripts/inspect_position.py --checkpoint checkpoints/residual_depth.pt \
  --board "0,0,1,0,0,1,0,0,0,5,0,0,1,0" --player 0 --simulations 50
```

This is a capture position from the tests. Check whether search gives visits to
the capturing move, and whether the trained network's prior already favours it
before any simulations have run.
