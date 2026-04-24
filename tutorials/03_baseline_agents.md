# 03. Baseline Agents

Before AlphaZero, we need baselines. Baselines answer the practical question:
"Is learning doing anything?"

This repo includes:

- `RandomAgent`: samples a legal move.
- `GreedyAgent`: maximizes immediate store margin.
- `MinimaxAgent`: searches a small tree with alpha-beta pruning.

## Evaluation

A match result is noisy, so we play many games and alternate who starts.

If agent A wins `w` out of `n` games, the observed win rate is:

```text
p_hat = w / n
```

This is not the true strength; it is an estimate. Later, we compare checkpoints
with the same arena machinery.

## Code

Read:

- `src/kalah_zero/agents.py`
- `src/kalah_zero/evaluate.py`

Run:

```bash
python scripts/evaluate.py --agent-a greedy --agent-b random --games 50
python scripts/evaluate.py --agent-a minimax --agent-b greedy --games 20
```

