# 03. Baseline Agents

## Goal

This lesson teaches why every learning project needs opponents that are simple,
measurable, and a little bit embarrassing. If AlphaZero cannot beat random or
greedy play, the training loop is not yet learning useful behavior.

An agent is any object implementing:

```python
class Agent(Protocol):
    def select_action(self, state: GameState) -> int:
        """Choose one legal action for the player to move."""
```

Mathematically, an agent is a policy:

\[
\pi(a \mid s),
\]

which assigns probabilities to actions. A deterministic agent is the special
case where all probability mass goes to one action.

## Random Agent

The random policy is:

\[
\pi_{\text{rand}}(a \mid s) =
\begin{cases}
\frac{1}{|\mathcal{A}(s)|}, & a \in \mathcal{A}(s), \\
0, & \text{otherwise}.
\end{cases}
\]

Code:

```python
@dataclass(slots=True)
class RandomAgent:
    rng: random.Random | None = None

    def select_action(self, state: GameState) -> int:
        rng = self.rng or random
        return rng.choice(state.legal_actions())
```

Random play is weak, but it is excellent for smoke tests because it explores
many legal transitions.

## Greedy Agent

The greedy agent chooses the move with the best immediate store margin:

\[
a^\* =
\arg\max_{a \in \mathcal{A}(s)}
\left[
\operatorname{store}_p(T(s,a)) -
\operatorname{store}_{1-p}(T(s,a))
\right].
\]

Code:

```python
def select_action(self, state: GameState) -> int:
    player = state.current_player
    return max(
        state.legal_actions(),
        key=lambda action: state.apply(action).normalized_store_margin(player),
    )
```

Greedy play often grabs captures and store moves, but it can miss long-term
traps.

## Minimax Agent

Minimax approximates \(V_p^\*(s)\) with a depth limit \(d\):

\[
\hat{V}_{p,d}(s) =
\begin{cases}
z_p(s), & \operatorname{terminal}(s), \\
h_p(s), & d = 0, \\
\max_a \hat{V}_{p,d-1}(T(s,a)), & \operatorname{player}(s)=p, \\
\min_a \hat{V}_{p,d-1}(T(s,a)), & \operatorname{player}(s)\ne p.
\end{cases}
\]

The heuristic \(h_p\) in this repo is:

\[
h_p(s) =
\frac{
\Delta_{\text{store}}(s) + 0.25\,\Delta_{\text{pit}}(s)
}{
\max(1,\operatorname{stones}(s))
}.
\]

## Comparing Agents

For \(n\) games, the observed win rate is:

\[
\hat{p} = \frac{w}{n}.
\]

The arena alternates starting positions so first-player advantage is less
misleading:

```python
for index in range(games):
    if index % 2 == 0:
        record = play_game(agent_a, agent_b, pits=pits, stones=stones)
    else:
        record = play_game(agent_b, agent_a, pits=pits, stones=stones)
```

## Practice

Run:

```bash
python scripts/evaluate.py --agent-a greedy --agent-b random --games 50
python scripts/evaluate.py --agent-a minimax --agent-b greedy --games 20
```

Write down what you expect before each run. Baselines are most useful when they
make you form hypotheses.

