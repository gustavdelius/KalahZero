# 07. Self-Play Training Loop

AlphaZero learns from games against itself.

At each move:

1. Run MCTS from the current state.
2. Store the state and normalized visit counts.
3. Pick an action from the visit distribution.
4. Continue until the game ends.

At the end, every stored position receives the final game outcome from the
perspective of the player who was to move there.

Each sample is:

```text
(s_t, pi_t, z_t)
```

where:

- `s_t` is the state.
- `pi_t` is the MCTS visit-count policy.
- `z_t` is the final outcome.

## Temperature

Early in the game, action selection samples from visit counts. Later it becomes
greedy. This gives variety without making endgames unnecessarily noisy.

## Code

Read `self_play_game` in `src/kalah_zero/train.py`.

Run:

```bash
python scripts/self_play.py --games 4 --simulations 25
```

