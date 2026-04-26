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

The current model uses three residual blocks, each of width 128. We could either use more blocks or make each
block wider.


## Batched MCTS Evaluations

The biggest engineering upgrade is batching:
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

The code keeps the first MCTS implementation in `src/kalah_zero/mcts.py`
unchanged for study. The faster implementation lives next to it in
`src/kalah_zero/batched_mcts.py`. During one search it collects up to $B$ leaf
states, evaluates those states together, then backs each result up through its
own path. Here $B$ is the evaluation batch size:

$$
B = \texttt{eval\_batch\_size}.
$$

The requested batch size should not be too large compared with the number of
simulations. If $B$ is large and the search only has $N$ simulations, the tree
is expanded in too few waves. For example, with $N=100$ and $B=32$, many leaves
are selected before the search has learned from earlier evaluations. The code
therefore caps the effective batch size at approximately:

$$
B_{\text{effective}}
=
\min(B, \lfloor \sqrt{N} \rfloor).
$$

This keeps CPU batching useful while preserving enough selection/evaluation
waves for the tree to react to new value estimates.

Use it during training with:

```bash
python scripts/train.py --games 300 --simulations 150 --epochs 1 \
  --batched-mcts --eval-batch-size 32 \
  --output checkpoints/overnight-batched.pt
```

If a resumed checkpoint was trained with batched MCTS and you want to turn it
off for comparison, pass:

```bash
python scripts/train.py --resume checkpoints/overnight-batched.pt --no-batched-mcts
```

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
