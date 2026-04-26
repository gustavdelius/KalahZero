# 05. From UCT to AlphaZero PUCT

## Goal

UCT means Upper Confidence bounds applied to Trees, the selection rule from the
previous lesson. PUCT is the AlphaZero-style version of that rule. The "P" is
often read as "predictor" or "prior" because the neural network predicts which
moves deserve attention before search has tried them much.

This lesson explains the key AlphaZero move: replacing random rollout guidance
with a neural network that supplies policy priors and value estimates. A rollout
is a simulated continuation of the game used to estimate who is likely to win.
A policy prior is a probability assigned to a move before the search has
gathered many visit counts for that move.

The network represents:

$$
f_\theta(s) = (P_\theta(s), v_\theta(s)).
$$

Here $\theta$ denotes the learnable parameters of the neural network — the
collection of all weights and biases that training adjusts. The subscript
$\theta$ is a reminder that the function's behaviour depends on those
parameters. $P_\theta(s)$ is a probability distribution over moves
(so $P_\theta(s,a)$ is the probability assigned to action $a$ from state $s$),
and $v_\theta(s) \in [-1,1]$ is the predicted outcome for the player to move.

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

During self-play, AlphaZero deliberately perturbs the root prior:

$$
P'(s,a) =
(1-\varepsilon)P(s,a) + \varepsilon\eta_a,
\qquad
\eta \sim \operatorname{Dirichlet}(\alpha).
$$

A Dirichlet distribution is a probability distribution over probability
vectors. Here it produces a random legal-move distribution $\eta$ that can be
mixed into the network's prior. This encourages opening diversity. Without it,
self-play can collapse into a small set of familiar games.

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

After $K$ simulations, AlphaZero does not train on the single chosen action.
It trains on visit counts:

$$
\pi(a \mid s) =
\frac{N(s,a)}{\sum_{b \in \mathcal{A}(s)} N(s,b)}.
$$

This $\pi$ is a stronger target than the raw network prior because it includes
lookahead.

```python
total_visits = sum(visits)
policy = [visit / total_visits if total_visits else 0.0 for visit in visits]
return SearchResult(root=root, visits=visits, policy=policy, value=root.mean_value)
```

## Practice

Run:

```bash
python scripts/inspect_position.py --simulations 200 --tree-depth 1
```

Then open `src/kalah_zero/mcts.py` and set `use_puct=False` in a small script.
Compare how UCT and PUCT distribute visits when priors are uniform.
