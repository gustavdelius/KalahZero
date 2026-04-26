# 02. Perfect-Information Games

## Goal

This lesson connects Kalah to game-tree search. You will learn what a value
function means before neural networks enter the story.

Kalah is a finite, deterministic, perfect-information game. That means:

$$
\text{next state} = T(s, a)
$$

is known exactly, and both players observe the same state.

## Game Trees

A game tree shows all possible futures from a position. The root is the current
state. Each edge is a legal action, and each child is the state that results
from that action. From a Kalah state $s$, the first layer of the tree is:

$$
s
\longrightarrow
\{T(s,a) \mid a \in \mathcal{A}(s)\}.
$$

If three actions are legal at the root, the tree begins like this:

```text
s
├── a=0 --> T(s,0)
│   ├── reply 0 --> ...
│   └── reply 1 --> ...
├── a=2 --> T(s,2)
│   ├── reply 0 --> ...
│   └── reply 4 --> ...
└── a=5 --> T(s,5)
    ├── reply 1 --> ...
    └── reply 3 --> ...
```

Every node is a state, and every edge is a move. A single game is just one path
through this tree:

$$
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2
\xrightarrow{a_2} \cdots \xrightarrow{a_{n-1}} s_n.
$$

At a terminal state $s_n$, the reward is $z_p(s_n)$ for player $p$.
The value of a nonterminal state is the outcome under some rule for choosing
future moves.

## The Value Function

How good is a position for player $p$? To answer that we need to assume
something about how both players will behave in the future. The strongest
possible assumption is **perfect play**: both players always choose the move
that is best for themselves.

Starting from that assumption we can define $V_p^*(s)$ — the value of state
$s$ for player $p$ under perfect play — by working backwards from the end
of the game:

- If the game is already over, the value is just the known reward: $V_p^*(s) = z_p(s)$.
- If it is player $p$'s turn, player $p$ will pick the move that leads to the
  highest value for themselves: $V_p^*(s) = \max_a V_p^*(T(s,a))$.
- If it is the opponent's turn, the opponent will pick the move that leads to
  the *lowest* value for player $p$: $V_p^*(s) = \min_a V_p^*(T(s,a))$.

Writing all three cases together:

$$
V_p^*(s) =
\begin{cases}
z_p(s), & \operatorname{terminal}(s), \\
\max_{a \in \mathcal{A}(s)} V_p^*(T(s,a)), & \operatorname{player}(s)=p, \\
\min_{a \in \mathcal{A}(s)} V_p^*(T(s,a)), & \operatorname{player}(s)\ne p.
\end{cases}
$$

The asterisk in $V_p^*$ is a conventional marker meaning "optimal" — the maximum value
achievable against a perfect opponent. The alternating max and min is why this
is called **minimax**: the current player maximises, the opponent minimises.

The three cases map directly to code in `MinimaxAgent._search`:

```python
def _search(self, state, depth, perspective, ...):
    if state.is_terminal():                        # terminal case: known reward
        return state.reward_for_player(perspective)
    if depth <= 0:
        return self._evaluate(state, perspective)  # depth limit: use heuristic

    if state.current_player == perspective:        # own turn: maximise
        value = -math.inf
        for action in state.legal_actions():
            value = max(value, self._search(state.apply(action), depth - 1, perspective, ...))
        return value

    value = math.inf                               # opponent's turn: minimise
    for action in state.legal_actions():
        value = min(value, self._search(state.apply(action), depth - 1, perspective, ...))
    return value
```

The `...` placeholders hide the alpha-beta pruning arguments, which are a
speed optimisation covered later. The core structure is just the three cases
above.

## Depth-Limited Minimax

The formula above defines perfect play, but computing it exactly would require
searching the full game tree. Instead, the code uses depth-limited minimax:

The number $d$ is the remaining search depth: how many more moves we are willing
to look ahead before stopping and using a heuristic. If $d=0$, we do not look at
any children and simply evaluate the current state with $h_p(s)$. If $d=1$, we
try each legal move once and evaluate the resulting states. If $d=2$, we also
look at the replies to those moves.

$$
\hat{V}_{p,d}(s) =
\begin{cases}
z_p(s), & \operatorname{terminal}(s), \\
h_p(s), & d = 0, \\
\max_{a \in \mathcal{A}(s)} \hat{V}_{p,d-1}(T(s,a)),
& \operatorname{player}(s)=p, \\
\min_{a \in \mathcal{A}(s)} \hat{V}_{p,d-1}(T(s,a)),
& \operatorname{player}(s)\ne p.
\end{cases}
$$

The new symbol $h_p(s)$ is a heuristic evaluation function. It answers:
"if I stop searching here, how good does this position look for player $p$?"

In `MinimaxAgent`, the heuristic is:

$$
h_p(s)
=
\frac{
\Delta_{\text{store}}(s)
+ 0.25\,\Delta_{\text{pit}}(s)
}{
\max(1,\operatorname{stones}(s))
}.
$$

where

$$
\Delta_{\text{store}}(s)
=
\operatorname{store}_p(s)
-
\operatorname{store}_{1-p}(s),
$$

and

$$
\Delta_{\text{pit}}(s)
=
\sum_{i \in P_p} b_i
-
\sum_{i \in P_{1-p}} b_i.
$$

Code:

```python
def _evaluate(self, state: GameState, perspective: int) -> float:
    """Heuristic score when the depth limit is reached."""
    own_pits = sum(state.pits_for(perspective))
    other_pits = sum(state.pits_for(1 - perspective))
    store_margin = state.store_for(perspective) - state.store_for(1 - perspective)
    pit_margin = own_pits - other_pits
    return (store_margin + 0.25 * pit_margin) / max(1, state.total_stones)
```

The store margin matters more because store stones are already secured. Pit
stones still matter, but they can later be captured or swept.

## How To Read The Recursion

Suppose player $p$ is the player for whom we are evaluating the root. The
algorithm carries that player around as `perspective`.

At every node:

- If the game is over, return the true reward $z_p(s)$.
- If the depth limit is reached, return the heuristic $h_p(s)$.
- If `state.current_player == perspective`, choose the maximum child value.
- Otherwise, choose the minimum child value.

The top-level action choice is:

$$
a^*
=
\arg\max_{a \in \mathcal{A}(s)}
\hat{V}_{p,d-1}(T(s,a)).
$$

Here $\arg\max$ means "the argument that maximises": $a^*$ is the specific
action $a$ for which $\hat{V}_{p,d-1}(T(s,a))$ is largest.

That is exactly what `select_action` does:

```python
def select_action(self, state: GameState) -> int:
    """Return the action with the highest minimax value at the configured depth."""
    perspective = state.current_player
    best_action = state.legal_actions()[0]
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf
    for action in state.legal_actions():
        value = self._search(state.apply(action), self.depth - 1, perspective, alpha, beta)
        if value > best_value:
            best_action = action
            best_value = value
        alpha = max(alpha, best_value)
    return best_action
```

Notice that `perspective` does not change. Even when the opponent is to move,
we still ask: "how good is this position for the original player?"

## Extra Turns

Kalah has extra turns, so the player may not alternate after every move. In this
project, extra turns happen after own-store landings and after captures. The
formula above still works because it checks $\operatorname{player}(s)$ at the
new state instead of assuming alternation.

In code, minimax compares `state.current_player` with the original perspective:

```python
if state.current_player == perspective:
    value = -math.inf
    for action in state.legal_actions():
        value = max(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
    return value

value = math.inf
for action in state.legal_actions():
    value = min(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
return value
```

## Alpha-Beta Pruning

In plain minimax the number of nodes to explore to depth $d$ is roughly:

$$
1 + b + b^2 + \cdots + b^d
= \frac{b^{d+1}-1}{b-1}
= O(b^d),
$$

where $b$ is the branching factor (the number of legal moves per position;
note this is a different use of the letter $b$ from the board vector in the
previous lesson).

If $b=6$ and $d=8$, this is already more than $2$ million nodes in the
worst case. Alpha-beta pruning gives the same minimax answer while skipping
branches that cannot change the decision.

Alpha-beta pruning carries two bounds:

$$
\alpha =
\text{the best value the maximizing side has already found on this path},
$$

$$
\beta =
\text{the best value the minimizing side has already found on this path}.
$$

Think of them as promises from ancestors in the search tree:

- $\alpha$: "Max already has an option worth at least this much."
- $\beta$: "Min already has an option that can hold the value to at most this
  much."

If at any point

$$
\alpha \ge \beta,
$$

then the current branch cannot be chosen by rational play. It is safe to stop
searching it.

## Why The Prune Is Safe

Consider a minimizing node. Its job is:

$$
\min_a \hat{V}_{p,d-1}(T(s,a)).
$$

Suppose one child has value $0.2$. Then the minimizing player can already
force the value to be at most $0.2$, so $\beta = 0.2$.

Now suppose an ancestor maximizing node already has another branch worth
$\alpha = 0.5$. The maximizing player will never choose a branch where the
opponent can hold the result to $0.2$. So if

$$
\beta \le \alpha,
$$

the rest of the minimizing node's children cannot matter.

The maximizing case is symmetric: once max finds a value at least as large as
the current $\beta$, min will avoid the branch.

## Alpha-Beta In Code

At a maximizing node, the code raises $\alpha$:

```python
value = -math.inf
for action in state.legal_actions():
    value = max(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
    alpha = max(alpha, value)
    if alpha >= beta:
        break
return value
```

At a minimizing node, the code lowers $\beta$:

```python
value = math.inf
for action in state.legal_actions():
    value = min(value, self._search(state.apply(action), depth - 1, perspective, alpha, beta))
    beta = min(beta, value)
    if alpha >= beta:
        break
return value
```

The `break` is the prune. It does not approximate. It skips work that cannot
change the exact depth-limited minimax result.

## A Tiny Worked Example

Imagine a root maximizing node with two candidate moves, $A$ and $B$.

Move $A$ has already been searched and gives:

$$
\hat{V}(A) = 0.4.
$$

So the root has:

$$
\alpha = 0.4.
$$

Now search move $B$, which leads to a minimizing node. The first reply by the
opponent gives:

$$
\hat{V}(B_1) = 0.1.
$$

Because the opponent is minimizing, the value of move $B$ is now known to be
at most $0.1$:

$$
\beta = 0.1.
$$

Since

$$
\alpha = 0.4 \ge 0.1 = \beta,
$$

the root player will prefer $A$ no matter what the remaining replies under
$B$ are. The rest of move $B$'s subtree can be pruned.

## Move Ordering

Alpha-beta pruning is strongest when good moves are searched first. If the best
moves appear early, $\alpha$ and $\beta$ tighten quickly, and more branches
are cut.

This teaching implementation keeps move ordering simple so the algorithm is
easy to read. A stronger minimax baseline could order actions by a greedy score
before searching them.

The final result is still depth-limited minimax; move ordering changes speed,
not the returned value.

## Why This Is Not AlphaZero Yet

Minimax asks:

$$
\text{"What happens if I search this tree deeply enough?"}
$$

AlphaZero asks:

$$
\text{"Can a neural network guide search, and can search train the network?"}
$$

Minimax gives us the conceptual baseline: states have values, moves can be
ranked, and exact search becomes expensive.

## Practice

Run:

```bash
python scripts/evaluate.py --agent-a minimax --agent-b random --games 10
```

Then change the minimax depth in `src/kalah_zero/agents.py` or pass a smaller
depth in a small script. Predict how speed and strength should change.
