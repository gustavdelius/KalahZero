# 04. Monte Carlo Tree Search

## Goal

This lesson introduces Monte Carlo Tree Search, the search procedure that
AlphaZero modifies rather than replaces.

Monte Carlo Tree Search is abbreviated as MCTS. "Monte Carlo" means it uses
sampled trials or simulations, and "tree search" means it builds part of the
game tree instead of expanding the whole thing.

Minimax tries to evaluate every branch up to a fixed depth. MCTS instead spends
more effort on moves that look promising.

## Node Statistics

For each edge $(s,a)$, MCTS stores:

$$
N(s,a) = \text{number of visits},
$$

$$
W(s,a) = \text{sum of backed-up values},
$$

$$
Q(s,a) = \frac{W(s,a)}{N(s,a)}.
$$

In code, each child node stores these quantities:

```python
@dataclass(slots=True)
class SearchNode:
    state: GameState
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

The `prior` field in this code is a move probability supplied before search has
many visits. Plain UCT does not need it, but the next lesson's PUCT rule does.

Note that the action $a$ is not stored inside `SearchNode`. In a tree, each
child is reachable by exactly one action from its parent, so the action is
stored as the key in the parent's children dictionary (for example,
`children: dict[int, SearchNode]`). Putting the action inside the child node
would be redundant.

## UCT Selection

UCT stands for Upper Confidence bounds applied to Trees. It is the rule that
decides which child node to visit next. Classic UCT chooses the child with the
largest score:

$$
\operatorname{UCT}(s,a) =
Q(s,a)
+ c
\sqrt{
\frac{\log(N(s)+2)}{1 + N(s,a)}
}.
$$

Here $N(s) = \sum_{a \in \mathcal{A}(s)} N(s,a)$ is the total number of times
the parent state $s$ has been visited across all simulations. It appears in
the numerator so that the exploration bonus grows slowly as the total search
budget increases, encouraging the agent to keep revisiting uncertain moves.

The exploitation term $Q(s,a)$ prefers moves that have worked. The
exploration term is large when $N(s,a)$ is small.

## Where UCT Comes From

UCT adapts an idea from a simpler problem called the multi-armed bandit problem.
Imagine you are choosing between several slot-machine arms. Each arm has an
unknown average reward. You want to earn high reward, but you also need to try
uncertain arms enough times to learn whether they are good.

For one arm $a$, suppose we have sampled it $N(a)$ times and observed rewards
$X_1,\ldots,X_{N(a)}$ in the range $[-1,1]$. Its empirical mean is:

$$
\bar{X}_a =
\frac{1}{N(a)}
\sum_{i=1}^{N(a)} X_i.
$$

This is our best estimate of the arm's true mean reward, but it is uncertain.
When $N(a)$ is small, the estimate might be wrong by a lot. When $N(a)$ is
large, the estimate is more reliable.

A concentration inequality such as Hoeffding's inequality says, informally,
that the true mean is unlikely to be much larger than the empirical mean:

$$
\mu_a
\lesssim
\bar{X}_a
+
\sqrt{\frac{\log n}{N(a)}},
$$

where $n$ is the total number of samples taken across all arms. The square-root
term is an uncertainty bonus. It shrinks as $N(a)$ grows and grows slowly as
the total search budget $n$ grows.

This gives the Upper Confidence Bound idea:

$$
\operatorname{UCB}(a)
=
\bar{X}_a
+
c
\sqrt{\frac{\log n}{N(a)}}.
$$

The name "upper confidence bound" comes from reading the score as an optimistic
estimate: not "how good has this arm looked so far?" but "how good could this
arm plausibly be?"

MCTS applies this bandit rule locally at each tree node. At a game state $s$:

- the "arms" are legal actions $a \in \mathcal{A}(s)$,
- the empirical mean $\bar{X}_a$ becomes the search value $Q(s,a)$,
- the arm visit count $N(a)$ becomes $N(s,a)$,
- the total samples $n$ become the parent visit count $N(s)$.

So the tree version is:

$$
Q(s,a)
+
c
\sqrt{
\frac{\log N(s)}{N(s,a)}
}.
$$

The implementation uses $N(s)+2$ and $1+N(s,a)$:

$$
Q(s,a)
+
c
\sqrt{
\frac{\log(N(s)+2)}{1+N(s,a)}
}.
$$

Those small offsets avoid division by zero and keep the logarithm defined during
the first few visits. They do not change the main idea: choose the action with
the largest optimistic value.

The constant $c$ controls the exploration-exploitation tradeoff:

- larger $c$ means more exploration of uncertain moves,
- smaller $c$ means more exploitation of moves whose $Q$ value is already high.

UCT is not a proof that every individual search decision is correct. It is a
principled way to allocate simulations so that obviously bad moves receive less
attention while uncertain moves still get chances to prove themselves.

The implementation can switch between UCT and PUCT. PUCT is the AlphaZero-style
variant introduced in the next lesson; for now, it is enough to know that PUCT
adds a learned policy prior to the UCT idea.

```python
if self.use_puct:
    exploration = self.c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (1 + child.visit_count)
else:
    exploration = self.c_puct * math.sqrt(
        math.log(parent.visit_count + 2) / (1 + child.visit_count)
    )
return q + exploration
```

## The Four Phases

One simulation consists of:

$$
\text{selection} \rightarrow \text{expansion} \rightarrow
\text{evaluation} \rightarrow \text{backup}.
$$

The code mirrors that sequence:

```python
for _ in range(self.simulations):
    path = self._select_path(root)
    leaf = path[-1]
    if leaf.state.is_terminal():
        value = leaf.state.reward_for_player(leaf.state.current_player)
    else:
        _, value = self._expand(leaf, evaluator)
    self._backup(path, value)
```

## Backup and Player Perspective

Kalah is a two-player zero-sum game: every stone gained by one player is lost
by the other, so what is good for one player is equally bad for the other.
That means if a position has value $+0.8$ for the player to move, it has value
$-0.8$ for the opponent.

A leaf value is always from the leaf state's current player's perspective. When
the path crosses to the other player, the sign must flip:

$$
V_{1-p}(s) = -V_p(s)
$$

for win/loss rewards. Kalah extra turns, including capture turns in this
variant, mean the sign flips only when the player-to-move changes.

```python
if node.state.current_player == child_player:
    node_value = value_for_child
else:
    node_value = -value_for_child
```

## Practice

Run:

```bash
python scripts/inspect_position.py --simulations 100 --tree-depth 2
```

Then rerun with `--simulations 10` and `--simulations 500`. Watch how the visit
distribution becomes less noisy.
