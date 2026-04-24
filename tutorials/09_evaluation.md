# 09. Evaluation and Elo-Style Comparisons

Training loss is not playing strength. We evaluate by games.

For two agents A and B:

```text
score_A = wins_A + 0.5 * draws
rate_A = score_A / games
```

This repo reports simple win/draw/loss counts. That is enough for early
learning. Elo-style ratings can be added later by converting match results into
rating updates.

## Confidence

Small match counts lie. If an agent wins 6 out of 10, that is encouraging but
not decisive. If it wins 120 out of 200, the evidence is much stronger.

## Code

Read `src/kalah_zero/evaluate.py`.

Run:

```bash
python scripts/evaluate.py --agent-a mcts --agent-b greedy --games 40
```

