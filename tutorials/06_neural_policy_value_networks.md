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
\frac{o_0}{P},\ldots,\frac{o_{m-1}}{P},
\frac{r_{m-1}}{P},\ldots,\frac{r_0}{P},
\frac{O-R}{B},
\frac{T}{C},
1
\right],
$$

where:

- $o_i$ are the current player's pit stones, ordered from pit $0$ to pit $m-1$.
- $r_i$ are the opponent's pit stones, listed in **reverse** order
  ($r_0$ corresponds to the opponent's pit $m-1$, $r_{m-1}$ to pit $0$).
  This reversal makes the encoding symmetric: pit $i$ for the current player
  sits opposite pit $i$ for the opponent in the input vector, just as it does
  physically on the board.
- $O$ is the current player's store.
- $R$ is the opponent's store.
- $O-R$ is the store margin from the current player's perspective.
- $T$ is the total number of stones still represented on the board and in the
  stores.
- $P$ is a fixed pit scale. In the code, $P=18$.
- $B$ is a fixed store-margin scale. In the code, $B=72$.
- $C$ is a fixed total-stone scale. In the code, $C=72$.

The scales are fixed constants, that transform the
network inputs into a comfortable numeric range.

The two store counts are not encoded separately. For choosing a move, adding
the same number of stones to both stores does not change the store advantage.
What matters is the margin:

$$
M = O - R.
$$

The total number of stones $T$ is included separately because positions with
different starting stone counts can have different dynamics even when their
relative pit pattern is similar.

Code:

```python
PIT_STONE_SCALE = 18.0
STORE_STONE_SCALE = 72.0
TOTAL_STONE_SCALE = 72.0


def encode_features(state: GameState) -> list[float]:
    player = state.current_player
    opponent = 1 - player
    own = [stones / PIT_STONE_SCALE for stones in state.pits_for(player)]
    other = [stones / PIT_STONE_SCALE for stones in reversed(state.pits_for(opponent))]
    store_margin = (state.store_for(player) - state.store_for(opponent)) / STORE_STONE_SCALE
    total_stones = state.total_stones / TOTAL_STONE_SCALE
    return own + other + [store_margin, total_stones, 1.0]
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


## Residual MLPs

The current network is an MLP, short for multilayer perceptron. In this repo
that means a feed-forward network built from linear layers and nonlinearities:

$$
x
\rightarrow h_1
\rightarrow h_2
\rightarrow (p,v).
$$

Each layer must learn not
only useful new features, but also how to preserve information that should pass
through unchanged.

Thus each layer is a residual block that learn a correction to the
current representation. If the input to a block is $h$, the block computes:

$$
\operatorname{Block}(h)
=
h + F_\phi(h),
$$

where $F_\phi$ is a small neural network with its own learnable parameters
$\phi$ (the weights of the two linear layers inside the block), often two
linear layers with a ReLU between them:

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

This is still an MLP. There is no attention mechanism and no convolution. 

As explained in the previous lesson, evaluator masks illegal moves before search sees probabilities.
Masking means setting illegal moves to probability zero:

```python
mask = torch.full_like(logits, float("-inf"))
for action in state.legal_actions():
    mask[action] = 0.0
probs = torch.softmax(logits + mask, dim=0)
```


## Practice

Run:

```bash
python -m pytest tests/test_encoding_training.py
python scripts/train.py --games 2 --simulations 10 --epochs 1
```

Then print `encode_features(GameState.new_game())` and identify each feature.
