# 08. AlphaZero Loss Derivation

For one training example `(s, pi, z)`, the network predicts:

```text
(p, v) = f_theta(s)
```

The loss has three terms:

```text
L = (z - v)^2 - sum_a pi_a log p_a + lambda ||theta||^2
```

## Value Loss

```text
(z - v)^2
```

This trains the network to predict the final outcome.

## Policy Loss

```text
- sum_a pi_a log p_a
```

This is cross-entropy from the search policy `pi` to the network policy `p`.
The target is not the raw move played; it is the stronger distribution produced
by search.

## Regularization

```text
lambda ||theta||^2
```

This discourages unnecessarily large weights.

## Code

Read `train_step` in `src/kalah_zero/train.py`.

