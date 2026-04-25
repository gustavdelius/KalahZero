# 07. Self-Play Training Loop

## Goal

This lesson explains how AlphaZero creates its own dataset. There is no database
of expert Kalah games. The system improves by playing itself, searching, and
training on the search results.

Self-play means the current agent plays both sides. The games it generates are
then used as training data for the next version of the same agent.

One self-play game produces samples:

$$
\mathcal{D}_{\text{game}} =
\{(s_t, \pi_t, z_t)\}_{t=0}^{T-1}.
$$

## At Each Move

At time $t$:

1. Run Monte Carlo Tree Search (MCTS) from $s_t$.
2. Convert visit counts into a policy target $\pi_t$.
3. Pick an action $a_t$.
4. Apply the action to get $s_{t+1}=T(s_t,a_t)$.

The code is direct:

```python
while not state.is_terminal():
    result = mcts.search(state, evaluator)
    temperature = 1.0 if move_index < config.temperature_moves else 0.0
    action = result.select_action(temperature=temperature, rng=rng)
    trajectory.append((state, tuple(result.policy), state.current_player))
    state = state.apply(action)
    move_index += 1
```

## Visit Counts Become Targets

The target policy is:

$$
\pi_t(a) =
\frac{N(s_t,a)^{1/\tau}}
{\sum_b N(s_t,b)^{1/\tau}},
$$

where $\tau$ is the temperature. Temperature is a knob that controls how random
the action choice is. In the code, temperature is used when selecting the played
action. The stored policy remains the normalized visit distribution returned by
search.

When $\tau = 1$, actions are sampled roughly according to visits. As
$\tau \to 0$, selection becomes greedy:

$$
a_t = \arg\max_a N(s_t,a).
$$

## Final Outcome Becomes Value Target

After the terminal state $s_T$, each earlier position receives:

$$
z_t = z_{p_t}(s_T),
$$

where $p_t$ is the player to move at $s_t$.

Code:

```python
return [
    TrainingSample(position, policy, state.reward_for_player(player))
    for position, policy, player in trajectory
]
```

This line is easy to miss. The value target is not "did player 0 win?" It is
"did the player who was about to move in this position eventually win?"

## Replay Buffer

The replay buffer is a fixed-size memory of recent training samples. It
approximates a dataset:

$$
\mathcal{D} =
\bigcup_i \mathcal{D}_{\text{game }i}.
$$

It has finite capacity, so old samples are dropped:

```python
def add_many(self, samples: list[TrainingSample]) -> None:
    self.samples.extend(samples)
    overflow = len(self.samples) - self.capacity
    if overflow > 0:
        del self.samples[:overflow]
```

## Safe Interruption And Resume

Long CPU runs should be interruptible. CPU means central processing unit: the
ordinary laptop processor, as opposed to a GPU, a graphics processing unit. The
training script periodically saves a checkpoint, which is a file containing
enough state to resume later:

- model weights,
- optimizer state,
- replay samples,
- random-number generator state,
- the training config,
- and the number of completed self-play games.

If you press `Ctrl+C`, the script saves before exiting. Resume with:

```bash
python scripts/train.py --resume checkpoints/overnight.pt
```

On resume, `--games` means the total number of games you want completed. For
example, if a checkpoint has already completed 120 games, then `--games 300`
continues from game 121 and stops after game 300.

## Practice

Run:

```bash
python scripts/self_play.py --games 4 --simulations 25
```

Then reduce `--simulations` to `1`. Explain why the generated targets are much
less informative.
