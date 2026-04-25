# 09. Evaluation and Elo-Style Comparisons

## Goal

This lesson separates "the loss went down" from "the agent got stronger."
AlphaZero-style systems must be evaluated by playing games.

Elo is a rating system originally designed for chess. It converts match results
into numbers that estimate relative player strength.

Suppose agent $A$ plays $n$ games and gets $w$ wins, $d$ draws, and
$\ell$ losses. Its score rate is:

$$
\hat{s}_A =
\frac{w + \frac{1}{2}d}{n}.
$$

The current `ArenaResult` reports wins, losses, and draws directly:

```python
@dataclass(frozen=True, slots=True)
class ArenaResult:
    games: int
    wins_0: int
    wins_1: int
    draws: int

    @property
    def win_rate_0(self) -> float:
        return self.wins_0 / max(1, self.games)
```

## Alternating First Player

Kalah may have a first-player advantage. To avoid confusing "agent strength"
with "started first," the arena swaps seats:

```python
if index % 2 == 0:
    record = play_game(agent_a, agent_b, pits=pits, stones=stones)
    winner_is_a = record.winner == 0
else:
    record = play_game(agent_b, agent_a, pits=pits, stones=stones)
    winner_is_a = record.winner == 1
```

Mathematically, this estimates:

$$
\frac{1}{2}
\Pr(A \text{ beats } B \mid A \text{ starts})
+
\frac{1}{2}
\Pr(A \text{ beats } B \mid B \text{ starts}).
$$

## Confidence

A win rate is an estimate. A Bernoulli trial is a random experiment with two
outcomes, such as win/loss if we ignore draws. If we model each game this way
with win probability $p$, then the standard error of $\hat{p}$ is approximately:

$$
\operatorname{SE}(\hat{p})
=
\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}.
$$

With $n=10$, this is large. With $n=200$, it is much smaller. This is why
tiny evaluations are good for smoke tests but poor for scientific claims.

## Elo Intuition

An Elo model predicts:

$$
\Pr(A \text{ beats } B)
=
\frac{1}{1 + 10^{(R_B - R_A)/400}},
$$

where $R_A$ and $R_B$ are ratings. This repo does not need full Elo yet,
but the formula explains why repeated arena matches can be turned into a rating
system later.

## Checkpoint Evaluation

The evaluation script supports both built-in agents and neural checkpoints:

```bash
python scripts/evaluate.py --agent-a greedy --agent-b random --games 50
python scripts/evaluate.py --checkpoint-a checkpoints/kalah_zero.pt --agent-b greedy --games 40
```

The checkpoint path creates a `NeuralEvaluator`, wraps it in Monte Carlo Tree
Search (MCTS), and plays it like any other agent.

## Noisy Opponents

Deterministic baselines answer one question: can agent $A$ beat this fixed
policy? Human opponents answer a slightly different question because they make
occasional mistakes. The evaluation script therefore includes noisy versions of
the main baselines:

```bash
python scripts/evaluate.py --checkpoint-a checkpoints/overnight.pt \
  --agent-b noisy-minimax --games 200 --noise-prob 0.05
```

The noisy wrapper uses an error probability $\epsilon$. At each move:

$$
a =
\begin{cases}
\text{a random legal action}, & \text{with probability } \epsilon, \\
\text{the base agent's action}, & \text{with probability } 1-\epsilon.
\end{cases}
$$

This does not replace deterministic evaluation. It complements it. A useful
suite compares against both `minimax` and `noisy-minimax`, often with random
openings:

```bash
python scripts/evaluate.py --checkpoint-a checkpoints/overnight.pt \
  --agent-b noisy-minimax --games 200 --simulations 100 \
  --opening-plies 4 --noise-prob 0.1
```

## Practice

Run:

```bash
python scripts/evaluate.py --agent-a greedy --agent-b random --games 10
python scripts/evaluate.py --agent-a greedy --agent-b random --games 100
```

Compare how stable the result feels.
