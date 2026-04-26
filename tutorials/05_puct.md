# 05. From UCT to AlphaZero PUCT

## Goal

UCT means Upper Confidence bounds applied to Trees, the selection rule from the
previous lesson. PUCT is the AlphaZero-style version of that rule. The "P" is
often read as "predictor" or "prior" because the neural network predicts which
moves deserve attention before search has tried them much.

The previous lesson introduced the `Evaluator` interface: a component that
MCTS calls on each newly expanded leaf to obtain a policy (move probabilities)
and a value (expected outcome). The lesson showed three concrete evaluators —
uniform, rollout-based, and neural — and noted that MCTS does not care which
one is used. This lesson is about what changes when the evaluator is a trained
neural network, and why that change is significant enough to modify the UCT
selection rule itself.

With a `RolloutEvaluator`, the policy returned is always uniform — every legal
move gets equal probability, because the evaluator has no opinion about which
moves are better before simulating them. The value is estimated by playing out
the game randomly, which is slow and noisy.

A neural network replaces both of those weaknesses at once. Given a board
position, it returns in a single forward pass:

$$
f_\theta(s) = (p_\theta(\cdot \mid s), v_\theta(s)),
$$

where $\theta$ denotes the learnable parameters (all weights and biases that
training adjusts). $P_\theta(s)$ without an action argument denotes the 
probability distribution over the actions — a vector with one entry $p_\theta(a\mid s)$ for each legal move $a$. $v_\theta(s) \in [-1,1]$ is the network's prediction of the final game
outcome for the player who is about to move from state $s$ — that is,
the player who will now choose an action. A value near $+1$ means the network
expects that player to win; near $-1$ means it expects them to lose. The
prediction is made before any action is taken from $s$. Both outputs are available immediately, without any random
playouts. The policy prior is now informative — early in search it can
concentrate attention on moves the network considers promising, before visit
counts are large enough to be reliable on their own.

## From UCT To PUCT

UCT explores every action from scratch:

$$
Q(s,a) + c
\sqrt{
\frac{\log(N(s)+2)}{1 + N(s,a)}
}.
$$

AlphaZero uses a prior $P(s,a)$ from the network:

$$
\operatorname{PUCT}(s,a) =
Q(s,a)
+ c_{\text{puct}}
P(s,a)
\frac{\sqrt{N(s)+1}}{1 + N(s,a)}.
$$

If the network thinks a move is plausible, $P(s,a)$ is larger, so that move
gets explored earlier. Search can still override the network if the value
statistics become poor.

Code:

```python
exploration = (
    self.c_puct
    * child.prior
    * math.sqrt(parent.visit_count + 1)
    / (1 + child.visit_count)
)
return q + exploration
```

## Policy Masking

The network outputs one logit per pit, even when a pit is empty. Search must
mask illegal actions:

$$
\tilde{P}(s,a) =
\begin{cases}
\frac{\max(0,P(s,a))}{\sum_{b \in \mathcal{A}(s)} \max(0,P(s,b))},
& a \in \mathcal{A}(s), \\
0, & a \notin \mathcal{A}(s).
\end{cases}
$$

The implementation falls back to a uniform distribution if all legal priors are
zero:

```python
for action in legal:
    masked[action] = max(0.0, float(policy[action]))
total = sum(masked)
if total <= 0 and legal:
    for action in legal:
        masked[action] = 1.0 / len(legal)
```

This fallback is pedagogically important. Early in training, the network may be
bad; the search should remain well-defined.

## Root Dirichlet Noise

During self-play, AlphaZero deliberately perturbs the priors of the root
node's children:

$$
P'(s,a) =
(1-\varepsilon)P(s,a) + \varepsilon\eta_a,
\qquad
\eta \sim \operatorname{Dirichlet}(\alpha).
$$

This perturbation is applied only to the root — the node representing the
current game state — immediately after it is expanded, before any simulations
run. It modifies the priors of the root's direct children, which affects how
the PUCT formula scores those children during selection. Nodes deeper in the
tree are expanded normally and receive unperturbed priors from the network.
The noise therefore only influences which first move each simulation explores;
once a simulation descends past the root, it follows the unmodified network
priors.

A Dirichlet distribution is a probability distribution over probability
vectors. Here it produces a random legal-move distribution $\eta$ that can be
mixed into the network's prior. This encourages opening diversity: without it,
every self-play game would begin with the same first moves, causing the replay
buffer to fill with near-identical positions and starving the network of
variety.

Code:

```python
noise = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
total = sum(noise)
for action, sample in zip(actions, noise):
    child.prior = (
        (1.0 - self.dirichlet_epsilon) * child.prior
        + self.dirichlet_epsilon * sample / total
    )
```

## Search Policy

After $K$ simulations the root's children have accumulated visit counts
$N(s,a)$. These are normalised into a probability distribution:

$$
\pi(a \mid s) =
\frac{N(s,a)}{\sum_{b \in \mathcal{A}(s)} N(s,b)}.
$$

This distribution $\pi$ is the output of the search. It is richer than the
raw network prior $P_\theta(s,a)$: moves that looked good under lookahead
received many visits and so have high $\pi$, while moves that looked poor
received few visits regardless of what the network initially thought of them.

```python
total_visits = sum(visits)
policy = [visit / total_visits if total_visits else 0.0 for visit in visits]
return SearchResult(root=root, visits=visits, policy=policy, value=root.mean_value)
```

This `policy` field in `SearchResult` is what callers use to select the move
to play (via `select_action`). Its role as a training signal for the network
is discussed in lesson 07.

## Practice

Run:

```bash
python scripts/inspect_position.py --simulations 200 --tree-depth 1
```

Then open `src/kalah_zero/mcts.py` and set `use_puct=False` in a small script.
Compare how UCT and PUCT distribute visits when priors are uniform.
