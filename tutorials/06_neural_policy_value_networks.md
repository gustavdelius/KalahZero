# 06. Neural Policy and Value Networks

## Goal

This lesson explains what the neural network predicts, how a Kalah board becomes
a tensor, and why AlphaZero uses two heads. A tensor is the array object used by
PyTorch; in this lesson it is just a vector of numbers. A head is an output
branch of a neural network.

The network approximates:

$$
f_\theta(s) = (p_\theta(\cdot \mid s), v_\theta(s)).
$$

The policy $p_\theta$ suggests promising actions. The value $v_\theta$
estimates the final result for the current player.

## Canonical Encoding

The same physical board can be viewed from either player. To avoid training two
separate networks, we encode from the current player's perspective.

For $m$ pits per side:

$$
x(s) =
\left[
\frac{o_0}{S},\ldots,\frac{o_{m-1}}{S},
\frac{r_{m-1}}{S},\ldots,\frac{r_0}{S},
\frac{O}{S},
\frac{R}{S},
1
\right],
$$

where:

- $o_i$ are the current player's pit stones.
- $r_i$ are the opponent's pit stones.
- $O$ is the current player's store.
- $R$ is the opponent's store.
- $S$ is the total number of stones.

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

The final $1$ is a bias-like feature. Neural layers already have biases, but
including it makes the input vector explicit for learning and inspection.

## Two-Headed Network

The trunk computes shared features:

$$
h = g_\theta(x).
$$

The policy head produces logits. A logit is an unnormalized score; the softmax
function converts logits into probabilities:

$$
\ell = W_p h + b_p,
\qquad
p_\theta(a \mid s) =
\frac{e^{\ell_a}}{\sum_b e^{\ell_b}}.
$$

The value head produces:

$$
v_\theta(s) = \tanh(W_v h + b_v).
$$

The `tanh` keeps values in $[-1,1]$, matching the reward scale.

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

## Residual MLPs

The current network is an MLP, short for multilayer perceptron. In this repo
that means a feed-forward network built from linear layers and nonlinearities:

$$
x
\rightarrow h_1
\rightarrow h_2
\rightarrow (p,v).
$$

Making the MLP wider gives it more capacity. Making it deeper can also help, but
plain deep networks can become harder to optimize. Each layer must learn not
only useful new features, but also how to preserve information that should pass
through unchanged.

A residual block gives the network an easier option: learn a correction to the
current representation. If the input to a block is $h$, the block computes:

$$
\operatorname{Block}(h)
=
h + F_\phi(h),
$$

where $F_\phi$ is a small neural network, often two linear layers with a ReLU
between them:

$$
F_\phi(h)
=
W_2 \operatorname{ReLU}(W_1 h + b_1) + b_2.
$$

So the whole update is:

$$
h_{k+1}
=
h_k + W_2 \operatorname{ReLU}(W_1 h_k + b_1) + b_2.
$$

The important idea is the skip connection $h_k \rightarrow h_{k+1}$. If a block
is not useful yet, it can learn $F_\phi(h) \approx 0$, so the block is close to
the identity function:

$$
h_{k+1} \approx h_k.
$$

That makes it safer to stack several blocks. The network can keep useful
features and add tactical corrections for patterns such as captures, extra
turns, and endgame sweeps.

A residual MLP version of the trunk would look like this:

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.activation = nn.ReLU()

    def forward(self, h):
        return self.activation(h + self.layers(h))


class ResidualKalahNet(nn.Module):
    def __init__(self, pits: int = 6, hidden_size: int = 256, blocks: int = 3) -> None:
        super().__init__()
        self.pits = pits
        self.input_layer = nn.Sequential(
            nn.Linear(input_size(pits), hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(blocks)]
        )
        self.policy_head = nn.Linear(hidden_size, pits)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, x):
        features = self.blocks(self.input_layer(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)
```

This is still an MLP. There is no attention mechanism and no convolution. The
difference is that each hidden representation can flow around a block while the
block learns an additive refinement.

For Kalah, this is a natural next architecture before attention. The board is
small, so the main need is not long-range perception over a large grid. The main
need is a little more capacity for tactical interactions among pits, stores,
captures, and extra turns.

## Illegal Move Masking

The network can assign a large logit to an empty pit. That is not a bug by
itself; the evaluator masks illegal moves before search sees probabilities.
Masking means setting illegal moves to probability zero:

```python
mask = torch.full_like(logits, float("-inf"))
for action in state.legal_actions():
    mask[action] = 0.0
probs = torch.softmax(logits + mask, dim=0)
```

Mathematically:

$$
p_\theta^{\text{legal}}(a \mid s) =
\begin{cases}
\frac{e^{\ell_a}}{\sum_{b \in \mathcal{A}(s)} e^{\ell_b}},
& a \in \mathcal{A}(s), \\
0, & a \notin \mathcal{A}(s).
\end{cases}
$$

## Practice

Run:

```bash
python -m pytest tests/test_encoding_training.py
python scripts/train.py --games 2 --simulations 10 --epochs 1
```

Then print `encode_features(GameState.new_game())` and identify each feature.
