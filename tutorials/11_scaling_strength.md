# 11. Scaling Strength

The first implementation is built for understanding. To make it stronger, scale
in this order:

1. More MCTS simulations.
2. Larger replay buffer.
3. More self-play games per checkpoint.
4. Batched neural evaluations inside MCTS.
5. A residual network instead of a small MLP.
6. Parallel self-play workers.

## What Changes Mathematically?

The core equations do not change. More compute gives better estimates:

```text
Q(s, a) -> more accurate value estimate
pi(a | s) = N(s, a) / sum_b N(s, b)
```

The engineering challenge is throughput: evaluating one neural position at a
time becomes wasteful.

## Safe Next Refactor

The best next code change is batched MCTS evaluation. Keep the game API exactly
the same, but collect leaf states and evaluate them together with one model
forward pass.

