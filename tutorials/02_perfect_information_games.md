# 02. Perfect-Information Games

Kalah is deterministic, turn-based, and fully observable. In principle we can
solve it by expanding a game tree.

Each node is a state `s`. Each edge is an action `a`. Each leaf has a final
reward `z`.

## Value

The value of a state for player `p` is:

```text
V_p(s) = expected outcome for p from state s
```

With perfect play and no randomness, this becomes minimax:

```text
V_p(s) = max_a V_p(T(s, a))  if p is to move
V_p(s) = min_a V_p(T(s, a))  if opponent is to move
```

Extra turns matter: if a move leaves the same player to move, the next layer is
still a maximizing layer for that same player.

## Why Not Exhaustive Search?

Kalah has a modest branching factor, usually at most 6, but games can last many
moves. A depth `d` tree costs roughly:

```text
O(b^d)
```

Even `6^20` is too large for casual search.

## Code

`MinimaxAgent` in `src/kalah_zero/agents.py` implements depth-limited minimax
with alpha-beta pruning. At a depth limit it uses a simple heuristic based on
store and pit margins.

Run:

```bash
python scripts/evaluate.py --agent-a minimax --agent-b random --games 10
```

