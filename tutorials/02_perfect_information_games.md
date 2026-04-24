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

A game tree alternates between states and actions:

$$
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2
\xrightarrow{a_2} \cdots \xrightarrow{a_{n-1}} s_n.
$$

At a terminal state $s_n$, the reward is $z_p(s_n)$ for player $p$.
The value of a nonterminal state is the outcome under some rule for choosing
future moves.

Under perfect play, the value for player $p$ is:

$$
V_p^\*(s) =
\begin{cases}
z_p(s), & \operatorname{terminal}(s), \\
\max_{a \in \mathcal{A}(s)} V_p^\*(T(s,a)), & \operatorname{player}(s)=p, \\
\min_{a \in \mathcal{A}(s)} V_p^\*(T(s,a)), & \operatorname{player}(s)\ne p.
\end{cases}
$$

The max line says "I choose my best move." The min line says "my opponent
chooses the move worst for me."

## Extra Turns

Kalah has extra turns, so the player may not alternate after every move. The
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

Plain minimax expands roughly:

$$
1 + b + b^2 + \cdots + b^d
= \frac{b^{d+1}-1}{b-1}
= O(b^d),
$$

where $b$ is branching factor and $d$ is depth.

Alpha-beta pruning keeps bounds:

$$
\alpha = \text{best value guaranteed for the maximizing player},
$$

$$
\beta = \text{best value guaranteed for the minimizing player}.
$$

If $\alpha \ge \beta$, the remaining branch cannot affect the final decision,
so it can be skipped.

```python
alpha = max(alpha, value)
if alpha >= beta:
    break
```

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

