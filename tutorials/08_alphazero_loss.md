# 08. AlphaZero Loss Derivation

## Goal

This lesson derives the training objective. You will see why the network learns
from two targets at once: the final winner and the search policy.

For one sample:

$$
(s, \pi, z),
$$

the network predicts:

$$
f_\theta(s) = (p_\theta(\cdot \mid s), v_\theta(s)).
$$

The AlphaZero-style loss is:

$$
\mathcal{L}(\theta)
=
(z - v_\theta(s))^2
- \sum_a \pi(a \mid s)\log p_\theta(a \mid s)
+ \lambda \lVert \theta \rVert_2^2.
$$

## Value Loss

The value term is the mean squared error, the average squared difference between a
prediction and its target. For one sample:

$$
\mathcal{L}_{\text{value}}
=
(z - v_\theta(s))^2.
$$

If $z=1$ and $v_\theta(s)=0.2$, the loss contribution is:

$$
(1 - 0.2)^2 = 0.64.
$$

This teaches the network to predict who eventually wins.

## Policy Loss

The policy target $\pi$ comes from Monte Carlo Tree Search (MCTS) visit counts. The cross-entropy term is:

$$
\mathcal{L}_{\text{policy}}
=
- \sum_a \pi(a \mid s)\log p_\theta(a \mid s).
$$

This is the **cross-entropy** between the target distribution $\pi$ and the
network's distribution $p_\theta$. Its minimum is zero, achieved when
$p_\theta = \pi$ everywhere. Intuitively, it measures how many extra bits you
would need to encode samples from $\pi$ using a code designed for $p_\theta$.
Minimising it pushes the network's distribution towards the search-derived
target.


## Regularization

Regularization is an extra penalty that discourages overly large model weights.
Here the regularizer is an L2 penalty, meaning a sum of squared weights:

$$
\mathcal{L}_{\text{reg}}
=
\lambda \lVert \theta \rVert_2^2
=
\lambda \sum_i \theta_i^2.
$$

It discourages unnecessarily large weights. In this codebase the default is
$\lambda = 10^{-4}$, small enough that the regulariser barely affects well-fit
samples but significant enough to prevent individual weights from growing
arbitrarily large during long training runs.

## Batch Loss In Code

Rather than computing the loss on one sample at a time, training averages over
a **minibatch** of $B$ samples drawn randomly from the replay buffer:

$$
\mathcal{L}(\theta)
=
\frac{1}{B}
\sum_{i=1}^{B}
\left[
(z_i - v_\theta(s_i))^2
- \sum_a \pi_i(a)\log p_\theta(a \mid s_i)
\right]
+ \lambda \lVert \theta \rVert_2^2.
$$

Averaging over a batch rather than a single sample has two benefits. First, the
gradient estimate is less noisy: one game position might be unusual, but the
average over many positions tends to point in a reliable direction. Second, modern
hardware (GPUs) can process a batch of positions in parallel almost as fast as
it can process one, so the cost per sample drops considerably.

The choice of batch size $B$ is a practical tradeoff. A larger batch gives a
smoother gradient estimate, which can allow a higher learning rate and faster
convergence. But it also means more memory and more computation per step, and
beyond a certain size the extra smoothness yields diminishing returns. In this
codebase the default is $B = 64$, a common starting point for small networks
training on CPU. If training on a GPU you might increase it to $256$ or $512$.
If memory is tight, reduce it; learning still works with $B = 16$, just more
noisily.

The implementation computes a minibatch average:

```python
states = torch.stack([encode_state(sample.state) for sample in batch])
policy_targets = torch.tensor([sample.policy for sample in batch], dtype=torch.float32)
value_targets = torch.tensor([sample.value for sample in batch], dtype=torch.float32)

logits, values = model(states)
policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
value_loss = F.mse_loss(values, value_targets)
loss = policy_loss + value_loss + l2_loss
```

The network returns logits rather than probabilities. A logit is an
unnormalized score; `log_softmax` converts those scores into log probabilities:

$$
\log p_\theta(a \mid s)
=
\ell_a - \log\sum_b e^{\ell_b}.
$$

This is more numerically stable than applying `softmax` and then `log`.

## Practice

Run:

```bash
python scripts/train.py --games 4 --simulations 25 --epochs 2
```

Watch the policy and value losses separately. If the value loss drops while
policy loss does not, the network may be learning winners faster than move
preferences.
