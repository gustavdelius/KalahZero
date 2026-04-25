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
f_\theta(s) = (p_\theta, v_\theta).
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

The value term is mean squared error, the average squared difference between a
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

The policy target $\pi$ comes from Monte Carlo Tree Search (MCTS) visit counts,
not from a human. The cross-entropy term is:

$$
\mathcal{L}_{\text{policy}}
=
- \sum_a \pi(a \mid s)\log p_\theta(a \mid s).
$$

If search assigns most visits to action $2$, the network is trained to make
action $2$ more likely next time. Search is therefore a policy improvement
operator:

$$
p_\theta(\cdot \mid s)
\xrightarrow{\text{MCTS}}
\pi(\cdot \mid s)
\xrightarrow{\text{training}}
p_{\theta'}(\cdot \mid s).
$$

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

It discourages unnecessarily large weights.

## Batch Loss In Code

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
