# 05. From UCT to AlphaZero PUCT

AlphaZero replaces random rollouts with a neural network:

```text
(P(s), v(s)) = f_theta(s)
```

`P(s)` is a policy prior over actions. `v(s)` is the predicted value for the
current player.

PUCT selects:

```text
score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

The prior guides exploration toward moves the network thinks are plausible. The
visit counts still correct the network through search.

## Root Noise

During self-play, AlphaZero adds Dirichlet noise to the root prior:

```text
P'(s, a) = (1 - eps) P(s, a) + eps * eta_a
eta ~ Dirichlet(alpha)
```

This prevents early self-play from becoming too repetitive.

## Code

Read `src/kalah_zero/mcts.py`.

Important pieces:

- `SearchNode`: stores `prior`, `visit_count`, `value_sum`, and children.
- `MCTS.search`: runs simulations and returns visit-count targets.
- `SearchResult.policy`: normalized visit counts used as the training target.

