# 12. Reading the Whole Codebase

One AlphaZero move flows through the code like this:

```text
GameState
  -> MCTS.search
    -> evaluator.evaluate
    -> state.apply
    -> backup values
  -> SearchResult.policy
  -> SearchResult.select_action
```

## Files

- `game.py`: rules, legal moves, rewards, rendering.
- `encoding.py`: canonical current-player features.
- `agents.py`: random, greedy, minimax, and MCTS-backed agents.
- `mcts.py`: UCT/PUCT tree search.
- `network.py`: PyTorch policy-value model and evaluator.
- `train.py`: self-play samples, replay buffer, loss step.
- `evaluate.py`: arena matches.

## Suggested Reading Order

1. Read `GameState.apply`.
2. Run `python -m unittest tests.test_game`.
3. Read `MCTS.search`.
4. Run `scripts/inspect_position.py`.
5. Read `KalahNet.forward`.
6. Read `self_play_game`.
7. Read `train_step`.

After that, run a tiny training loop and inspect the checkpoint behavior against
the baselines.

