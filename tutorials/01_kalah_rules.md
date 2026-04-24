# 01. Kalah Rules and State Representation

AlphaZero needs a game API before it needs a neural network. The API must answer
four questions:

- What is the current state?
- Which actions are legal?
- What state follows from an action?
- Who won?

In this repo, `GameState` in `src/kalah_zero/game.py` is immutable. A move does
not edit the old state; it returns a new one.

## Board Coordinates

For 6 pits, the board tuple has 14 integers:

```text
0 1 2 3 4 5 | 6 | 7 8 9 10 11 12 | 13
P0 pits       P0  P1 pits           P1
              store                 store
```

Actions are local pit numbers `0..5` for whichever player is to move. Player 1's
action `0` maps to board index `7`.

## Transition Function

The transition function is deterministic:

```text
s' = T(s, a)
```

`state.apply(action)` implements `T`. It removes stones from the selected pit,
sows one by one, skips the opponent store, captures when the last stone lands in
an empty own pit opposite a non-empty opponent pit, and grants an extra turn when
the last stone lands in the mover's store.

## Terminal States and Rewards

The game ends when either side has no pit stones. Remaining stones are swept into
the other store. The reward is:

```text
z_p =  1 if player p has more store stones
z_p =  0 if stores are tied
z_p = -1 otherwise
```

This is what `reward_for_player(player)` returns.

## Run It

```bash
python scripts/play_cli.py --agent greedy
python -m unittest tests.test_game
```

