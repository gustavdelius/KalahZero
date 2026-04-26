# 07. Self-Play Training Loop

## Goal

This lesson explains how AlphaZero creates its own dataset. There is no database
of expert Kalah games. The system improves by playing itself, searching, and
training on the search results.

Self-play means the current agent plays both sides. The games it generates are
then used as training data for the next version of the same agent.

A single self-play game lasts $T$ moves. At every move $t \in \{0, 1, \ldots,
T-1\}$ the system records a training sample consisting of three things:

- $s_t$ — the board state before the move,
- $\pi_t$ — the search policy at that state: the visit-count distribution
  produced by MCTS, which says how promising each move looked after lookahead,
- $z_t$ — the final game outcome for the player who was to move at $s_t$
  (either $+1$, $0$, or $-1$). This is not known until the game ends and is
  filled in retrospectively.

Collecting all $T$ samples gives the dataset for one game:

$$
\mathcal{D}_{\text{game}} =
\{(s_t, \pi_t, z_t)\}_{t=0}^{T-1}.
$$

The network is then trained to make its own policy and value predictions match
these targets: its policy output should resemble $\pi_t$ and its value output
should resemble $z_t$. This is explained in detail in lessons 08 and 09.

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

## Visit Counts Become Policy Target

The target policy is:

$$
\pi_t(a) =
\frac{N(s_t,a)^{1/\tau}}
{\sum_b N(s_t,b)^{1/\tau}},
$$

where $\tau > 0$ is the **temperature**. It controls how random the action
choice is.

When $\tau = 1$ the formula reduces to plain visit-count proportions, so each
action is sampled with probability proportional to how many visits it received.
When $\tau < 1$ the exponent $1/\tau > 1$ amplifies differences: actions with
more visits get disproportionately higher probability. As $\tau \to 0$ the most
visited action gets essentially all the probability mass, which is equivalent to
choosing greedily:

$$
a_t = \arg\max_a N(s_t,a).
$$

In the code the temperature is set to $1$ for the first `temperature_moves`
moves of each game and then to $0$ (greedy) for the rest. Early in the game
this encourages variety in the training data — sampling proportionally to
visits means the agent occasionally tries second-best moves, producing
different game continuations. Late in the game greedy selection makes the
match outcome more decisive and the resulting value target cleaner.

In the code, temperature is used only when selecting the played action $a_t$.
The stored policy $\pi_t$ is always the raw normalised visit distribution
returned by search, regardless of temperature.

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

## Random Opening Self-Play

If every self-play game begins at the same initial board, the replay buffer can
become too focused on common openings. To broaden the training distribution, the
trainer can make $k$ random legal moves before search-based self-play begins:

$$
s_{\text{start}} = T(T(\cdots T(s_0,a_0),a_1)\cdots,a_{k-1}).
$$

The random opening moves are not stored as training samples. They only choose a
less familiar starting state. After that, the usual AlphaZero loop takes over:
search, store the visit-count policy, play a move, and later attach the final
outcome.

The number of random opening moves can be set with the `--opening-plies` argument to `train.py` and `evaluate.py`.

The opening length can also be sampled uniformly from a range. If
$k_{\min}=0$ and $k_{\max}=8$, then each self-play game chooses:

$$
k \sim \operatorname{Uniform}\{0,1,\ldots,8\}.
$$

Then the trainer applies $k$ random legal opening moves before search-based
self-play begins. This is set with `--opening-plies-min 0 --opening-plies-max 8`.


## Mixed Starting Stones

Kalah can be played with either 4, 5 or 6 stones in each pit at the start of the game.
This is set wtih the `--stones` argument to `train.py` and `evaluate.py`.

To teach one network to handle several starting positions, the trainer can also
sample the number of starting stones. For example:

$$
c \sim \operatorname{Uniform}\{4,5,6\}.
$$

Then each self-play game starts from a new Kalah board with $c$ stones in every
pit. This can be set with `--stones-min 4 --stones-max 6`.



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
