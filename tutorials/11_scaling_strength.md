# 11. Scaling Strength

## Goal

This lesson explains how to make the agent stronger without losing sight of the
core algorithm.

The first implementation prioritizes clarity. Strength comes from improving the
quality of the policy target:

$$
\pi(a \mid s) =
\frac{N(s,a)}{\sum_b N(s,b)}.
$$

Better visit counts usually mean better targets.

## More Simulations

If each search uses $K$ simulations, the approximate compute cost is:

$$
\operatorname{cost}
\approx
K \cdot \operatorname{cost}(\text{one simulation}).
$$

Increasing $K$ usually improves move quality but slows self-play. Try:

```bash
python scripts/inspect_position.py --simulations 25
python scripts/inspect_position.py --simulations 400
```

## Larger Replay Buffer

The replay buffer stores:

$$
\mathcal{D} = \{(s_i,\pi_i,z_i)\}_{i=1}^{M}.
$$

If $M$ is too small, the network overfits recent games. If $M$ is too large,
training may lag behind the current policy. The default is intentionally modest:

```python
@dataclass(frozen=True, slots=True)
class TrainConfig:
    replay_capacity: int = 10_000
```

## Bigger Networks

The current model is a small multilayer perceptron:

$$
x \rightarrow h_1 \rightarrow h_2 \rightarrow (p,v).
$$

A stronger version could use residual blocks:

$$
h_{k+1} = h_k + F_k(h_k).
$$

Residual connections make deeper networks easier to train, but they add
complexity. MLP stands for multilayer perceptron; it means a feed-forward neural
network made from ordinary linear layers and nonlinearities. For learning
AlphaZero, the MLP is the right first model.

## Batched MCTS Evaluations

MCTS means Monte Carlo Tree Search. The biggest engineering upgrade is batching:
processing several positions together instead of one at a time. Instead of
evaluating one leaf at a time:

$$
f_\theta(s_1), f_\theta(s_2), \ldots, f_\theta(s_B),
$$

evaluate a batch:

$$
f_\theta
\left(
\begin{bmatrix}
x(s_1) \\
x(s_2) \\
\vdots \\
x(s_B)
\end{bmatrix}
\right).
$$

This makes better use of vectorized CPU operations and GPUs. CPU means central
processing unit, the ordinary laptop processor. GPU means graphics processing
unit, a processor designed for many parallel numeric operations.

## Practice

Train two small checkpoints:

```bash
python scripts/train.py --games 4 --simulations 10 --epochs 1 --output checkpoints/small.pt
python scripts/train.py --games 4 --simulations 50 --epochs 1 --output checkpoints/larger_search.pt
```

Then compare:

```bash
python scripts/evaluate.py --checkpoint-a checkpoints/larger_search.pt --checkpoint-b checkpoints/small.pt --games 20
```
