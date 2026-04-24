# 06. Neural Policy and Value Networks

The network approximates:

```text
f_theta(s) = (p, v)
```

where `p` is a probability distribution over pits and `v` is a scalar in
`[-1, 1]`.

## Encoding

`encode_features(state)` uses the current player's perspective:

```text
[own pits, opponent pits reversed, own store, opponent store, bias]
```

This canonical view lets one network learn for both players.

## Heads

`KalahNet` has a shared trunk and two heads:

- policy head: 6 logits, one per pit
- value head: one `tanh` output

Illegal moves are masked by `NeuralEvaluator`, so the network can produce logits
for all pits while search only considers legal actions.

## Code

Read:

- `src/kalah_zero/encoding.py`
- `src/kalah_zero/network.py`

Run after installing dependencies:

```bash
python scripts/train.py --games 2 --simulations 10 --epochs 1
```

