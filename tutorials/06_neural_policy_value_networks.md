# 06. Neural Policy and Value Networks

## Goal

This lesson explains what the neural network predicts, how a Kalah board becomes
a tensor, and why AlphaZero uses two heads.

The network approximates:

\[
f_\theta(s) = (p_\theta(\cdot \mid s), v_\theta(s)).
\]

The policy \(p_\theta\) suggests promising actions. The value \(v_\theta\)
estimates the final result for the current player.

## Canonical Encoding

The same physical board can be viewed from either player. To avoid training two
separate networks, we encode from the current player's perspective.

For \(m\) pits per side:

\[
x(s) =
\left[
\frac{o_0}{S},\ldots,\frac{o_{m-1}}{S},
\frac{r_{m-1}}{S},\ldots,\frac{r_0}{S},
\frac{O}{S},
\frac{R}{S},
1
\right],
\]

where:

- \(o_i\) are the current player's pit stones.
- \(r_i\) are the opponent's pit stones.
- \(O\) is the current player's store.
- \(R\) is the opponent's store.
- \(S\) is the total number of stones.

Code:

```python
def encode_features(state: GameState) -> list[float]:
    player = state.current_player
    opponent = 1 - player
    scale = float(max(1, state.total_stones))
    own = [stones / scale for stones in state.pits_for(player)]
    other = [stones / scale for stones in reversed(state.pits_for(opponent))]
    stores = [state.store_for(player) / scale, state.store_for(opponent) / scale]
    return own + other + stores + [1.0]
```

The final \(1\) is a bias-like feature. Neural layers already have biases, but
including it makes the input vector explicit for learning and inspection.

## Two-Headed Network

The trunk computes shared features:

\[
h = g_\theta(x).
\]

The policy head produces logits:

\[
\ell = W_p h + b_p,
\qquad
p_\theta(a \mid s) =
\frac{e^{\ell_a}}{\sum_b e^{\ell_b}}.
\]

The value head produces:

\[
v_\theta(s) = \tanh(W_v h + b_v).
\]

The `tanh` keeps values in \([-1,1]\), matching the reward scale.

Code:

```python
class KalahNet(nn.Module):
    def __init__(self, pits: int = 6, hidden_size: int = 128) -> None:
        self.trunk = nn.Sequential(
            nn.Linear(input_size(pits), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, pits)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, x):
        features = self.trunk(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)
```

## Illegal Move Masking

The network can assign a large logit to an empty pit. That is not a bug by
itself; the evaluator masks illegal moves before search sees probabilities:

```python
mask = torch.full_like(logits, float("-inf"))
for action in state.legal_actions():
    mask[action] = 0.0
probs = torch.softmax(logits + mask, dim=0)
```

Mathematically:

\[
p_\theta^{\text{legal}}(a \mid s) =
\begin{cases}
\frac{e^{\ell_a}}{\sum_{b \in \mathcal{A}(s)} e^{\ell_b}},
& a \in \mathcal{A}(s), \\
0, & a \notin \mathcal{A}(s).
\end{cases}
\]

## Practice

Run:

```bash
python -m pytest tests/test_encoding_training.py
python scripts/train.py --games 2 --simulations 10 --epochs 1
```

Then print `encode_features(GameState.new_game())` and identify each feature.

