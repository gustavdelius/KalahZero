# 04. Monte Carlo Tree Search

## Goal

This lesson introduces Monte Carlo Tree Search, the search procedure that
AlphaZero modifies rather than replaces.

Monte Carlo Tree Search is abbreviated as MCTS. "Monte Carlo" means it uses
sampled trials or simulations, and "tree search" means it builds part of the
game tree instead of expanding the whole thing.

Minimax tries to evaluate every branch up to a fixed depth. MCTS instead spends
more effort on moves that look promising.

## How MCTS Works

When estimating the value of a state, MCTS builds a search tree incrementally, one simulation at a time, for a chosen number of simulations. Each
simulation is a single pass through three steps:

1. **Selection.** Starting from the root, follow the most promising child at
   each node until reaching a **leaf** — a node whose children have not yet
   been added to the tree.
2. **Expansion and evaluation.** Call the evaluator on the leaf node. The
   evaluator returns two things at once: a **value** for the leaf (how good
   this position looks for the player to move) and a **policy** over all
   legal moves (how promising each child is before it has been visited).
   The children are then created and added to the tree, each receiving its
   policy probability as its `prior`.
3. **Backup.** Carry the leaf's value back up through every node on the path
   to the root, updating each node's visit count and value sum.

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

After many simulations, nodes on good lines have been visited many times and
have high average values. Nodes on bad lines have been visited less. At the
end of search, the root's visit counts tell us which moves search considered
most promising — this distribution is used both to choose the move to play and
as a training target for the neural network.


## The Search Tree and Node Statistics

The search tree is a network of `SearchNode` objects linked by Python
references. The root of the search tree is the current game state — the position
the agent is facing right now, not the opening position. If the game has
reached move 15, the root is the board after 14 moves have been played.

Each node owns a dictionary mapping action numbers to child nodes:

```python
children: dict[int, SearchNode]
```

The root's `children` contains one entry per legal action that has already been explored from the
root. Each child has its own `children` for the moves explored one level
deeper, and so on. Following `root.children[2].children[5]` reaches the state
two moves deep: action 2 from the root, then action 5 from there.

The action leading to a child is stored as the dictionary *key*, not inside the child node itself.
Because each node has exactly one parent in a tree, the key already captures
the incoming action — duplicating it inside the child would be redundant.

The tree only exists for the duration of one `search()` call. After the agent
picks its move, the tree is discarded and a fresh one is built on the next turn.

Beyond the game state and children, each node also records statistics that
accumulate as simulations pass through it:

$$
N(s,a) = \text{how many simulations have passed through this child},
$$
$$
W(s,a) = \text{sum of values those simulations returned},
$$

Here $s$ is the state of the parent and $a$ is the action that led from the parent to the current node. From these statistics we can calculate a quality score:

$$
Q(s,a) = \frac{W(s,a)}{N(s,a)} = \text{mean value — how good action } a \text{ has looked so far}.
$$


```python
@dataclass(slots=True)
class SearchNode:
    """One node in the MCTS tree, representing a single game state."""
    state: GameState
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)  # keyed by local action number

    @property
    def mean_value(self) -> float:
        """Return the average backed-up value, or 0 if the node has never been visited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

The `prior` is a move probability supplied by the evaluator when the node is
first created, before search has accumulated any visit counts. Plain UCT does
not need it, but the next lesson's PUCT rule does.

## The Evaluator

The evaluator is the component that MCTS calls during expansion to assess a
leaf node. It takes a game state and returns two things:

$$
\text{evaluator}(s) = (\text{policy},\ \text{value}),
$$

where the **policy** is a probability distribution over the $m$ pit indices
(one entry per possible action, including illegal ones that will be masked
later) and the **value** is a single number in $[-1, 1]$ estimating how good
the position is for the player to move.

In code, any object that implements this interface can serve as an evaluator:

```python
class Evaluator(Protocol):
    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        """Return (policy, value) from the current player's perspective."""
```

This design means MCTS does not need to know what is inside the evaluator.
The search logic is the same whether the evaluator is a neural network, a
random-rollout estimate, or a simple uniform guess.

Three evaluators are available in this codebase:

- **`UniformEvaluator`** — assigns equal probability to every legal move and
  returns value $0$. Useful for testing MCTS mechanics before any network
  exists.
- **`RolloutEvaluator`** — estimates the value by playing the game out
  randomly several times and averaging the results. This is the classical
  pre-AlphaZero approach.
- **`NeuralEvaluator`** — runs the trained neural network. This is what the
  full AlphaZero system uses, and what the next lessons build towards.

## UCT Selection

UCT stands for Upper Confidence bounds applied to Trees. It is the rule that
decides which child node to select as the most promising at each leaf. Classic UCT chooses the child with the
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

The UCT formula and the evaluator's policy play entirely different roles and
should not be confused:

- The **evaluator's policy** $P(s,a)$ is consulted once, at expansion time,
  to set the `prior` of each newly created child node. It is the evaluator's
  initial opinion about which moves look promising before any visits.
- The **UCT selection rule** is applied at every node during every simulation's
  selection phase. It uses only $Q(s,a)$ and $N(s,a)$ — the statistics
  accumulated so far — and completely ignores the `prior`.

With plain UCT the policy only affects which priors are stored on child nodes,
but those priors are never read back. This means UCT treats the evaluator
purely as a value estimator and discards its move-probability output.
AlphaZero's PUCT rule, introduced in the next lesson, changes this: it
multiplies the exploration term by the prior $P(s,a)$, so moves the network
considers promising get a larger exploration bonus early in search.

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
