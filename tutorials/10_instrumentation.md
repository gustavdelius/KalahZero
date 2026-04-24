# 10. Instrumentation and Debugging

AlphaZero systems fail quietly. Useful questions:

- Are legal moves masked correctly?
- Do visit counts sum to the number of simulations?
- Are values flipped only when the player changes?
- Does self-play produce varied openings?
- Is the value head predicting the eventual winner?

## Tree Dumps

`MCTS.dump_tree` prints:

```text
N = visit count
Q = mean value
P = prior
```

This is the fastest way to see whether search is behaving sensibly.

## Position Inspection

Run:

```bash
python scripts/inspect_position.py --simulations 200 --tree-depth 2
```

For custom boards:

```bash
python scripts/inspect_position.py --board "4,4,4,4,4,4,0,4,4,4,4,4,4,0" --player 0
```

