# 01. Kalah Rules and State Representation

## Goal

By the end of this lesson, you should be able to explain exactly what a Kalah
state is, how an action changes it, and why an AlphaZero project starts with a
small, trustworthy game engine.

## The Game in Plain Language

Kalah is a two-player board game played with stones and two rows of small **pits**. Each player owns a row of pits and one larger cup called a
**store** (sometimes called a mancala). At the start of the game every pit
contains the same number of stones and the stores are empty.

On your turn you choose one of your own non-empty pits and pick up all its
stones. You then **sow** them one by one into the pits and stores going
counter-clockwise around the board, skipping your opponent's store. Two special
rules apply:

- **Extra turn.** If your last stone lands in your own store, or if your move
  results in a capture (see below), you get another turn immediately.
- **Capture.** If your last stone lands in one of your own pits that was
  previously empty, and the pit directly opposite contains stones, you capture
  all of those opposite stones plus your landing stone into your store.

The game ends when all pits on one side of the board are empty. Any stones
remaining on the other side go into that player's store. The player with more
stones in their store wins.

## Game states

AlphaZero does not learn from pixels here. It learns from a formal game model.
For Kalah, the state of the game is represented by a tuple:

$$
s = (b, p),
$$

where $b$ is the board vector and $p \in \{0, 1\}$ is the player to move.

For the default game, $6$ pits and $4$ stones per pit, the board has
$2 \cdot 6 + 2 = 14$ entries:

$$
b = (b_0, b_1, \ldots, b_{13}).
$$

The stores are:

$$
\operatorname{store}(0) = 6,
\qquad
\operatorname{store}(1) = 13.
$$

Player $0$'s pits are indices $0,\ldots,5$. Player $1$'s pits are
indices $7,\ldots,12$. We write $P_p$ for the set of pit indices belonging to
player $p$, so $P_0 = \{0,1,2,3,4,5\}$ and $P_1 = \{7,8,9,10,11,12\}$.
A move is represented as a local pit number $a \in \{0,\ldots,5\}$ for the
player to move.

The diagram below shows the full board for the 6-pit game. Stones sow
counter-clockwise, so Player 0 sows left-to-right through their own pits then
up through Player 1's pits, and Player 1 sows right-to-left through their own
pits then down through Player 0's pits. Each player skips the opponent's store.

```text
                          Player 1
              <--------------------------------
 +----------+----+----+----+----+----+----+----------+
 | P1 store | 12 | 11 | 10 |  9 |  8 |  7 | P0 store |
 |  (b_13)  |    |    |    |    |    |    |  (b_6)   |
 |          |  0 |  1 |  2 |  3 |  4 |  5 |          |
 +----------+----+----+----+----+----+----+----------+
              -------------------------------->
                          Player 0
```

Pit $i$ for Player 0 sits directly opposite pit $12-i$ for Player 1 (pit 0
faces pit 12, pit 5 faces pit 7). This opposite pairing matters for the
capture rule.

Here is the corresponding code:

```python
@dataclass(frozen=True, slots=True)
class GameState:
    board: tuple[int, ...]
    current_player: int
    pits: int = 6

    @classmethod
    def new_game(cls, pits: int = 6, stones: int = 4) -> GameState:
        board = [stones] * pits + [0] + [stones] * pits + [0]
        return cls(tuple(board), current_player=PLAYER_0, pits=pits)
```

The `frozen=True` argument to `@dataclass` makes every instance immutable: any
attempt to assign to a field after construction raises an error. This is the
right choice here because search algorithms reuse states constantly. If
the code could mutated a game state in place, one branch of the search tree could corrupt
another branch.

## Legal Actions

An **action** is a choice of which pit to sow from. It is represented as a
local pit number $a \in \{0, 1, 2, 3, 4, 5\}$, where $0$ is the player's
leftmost pit and $5$ is their rightmost. The word "local" means the number
always counts from the current player's own side, regardless of which player
is moving: action $2$ for Player 0 refers to board index $2$, while action $2$
for Player 1 refers to board index $9$.

A pit is legal to choose only if it is non-empty — you must have stones to sow.
The legal action set is therefore:

$$
\mathcal{A}(s) =
\{a \mid a \in \{0,\ldots,5\} \text{ and } b_{\text{index}(p,\,a)} > 0\},
$$

where $\text{index}(p, a)$ converts a local pit number to a board index for
player $p$. In code, the definition is deliberately plain:

```python
def legal_actions(self) -> list[int]:
    """Return the list of local action numbers the current player may choose."""
    if self.is_terminal():
        return []
    return [
        self.action_for_index(self.current_player, index)
        for index in self.pit_indices(self.current_player)
        if self.board[index] > 0
    ]
```

Reading it step by step:

1. `self.pit_indices(self.current_player)` returns the **board indices** for the
   current player's row of pits. For Player 0 that is `range(0, 6)` (indices
   0–5); for Player 1 it is `range(7, 13)` (indices 7–12).

2. `if self.board[index] > 0` filters out empty pits — you can only sow from a
   pit that has stones in it.

3. `self.action_for_index(self.current_player, index)` converts a board index
   back to a **local action number** (0–5). For Player 0 the local number equals
   the board index directly. For Player 1 it subtracts 7, so board index 7
   becomes action 0, index 8 becomes action 1, and so on.

The result is a list of local pit numbers the current player may legally choose.

## Transition Function

A **transition function** is a precise description of how the game moves from
one state to the next. Given the current state $s$ and a chosen action $a$, it
returns the state $s'$ that results:

$$
s' = T(s, a).
$$

Having an explicit $T$ is what makes the game a formal model rather than a
written rulebook. Any algorithm — minimax, MCTS, self-play — can ask "what
happens if I do this?" by calling $T$, without needing to know anything about
sowing or captures directly.

For Kalah, $T$ does six things:

1. Pick up all stones from the selected pit.
2. Sow them counter-clockwise.
3. Skip the opponent's store.
4. Give the player another move if the last stone lands in their own store or
   the move makes a capture.
5. Capture when the last stone lands in an empty own pit opposite stones.
6. Sweep remaining stones if either side becomes empty.

The public API is small:

```python
def apply(self, action: int) -> GameState:
    """Return the state that results from the current player choosing `action`."""
    if self.is_terminal():
        raise ValueError("cannot apply an action to a terminal state")
    if action not in self.legal_actions():
        raise ValueError(f"illegal action {action}")
    ...
    return GameState(tuple(board), current_player=next_player, pits=self.pits)
```

The important design point is that all later algorithms see Kalah only through
this transition function. Minimax, Monte Carlo Tree Search (MCTS), and self-play
do not need special knowledge of sowing or capture.

## Extra Moves

This project uses a Kalah variant with two extra-move cases. Player $p$ moves
again if either:

- the final sown stone lands in player $p$'s own store, or
- the move captures stones from the opposite pit.

$$
\operatorname{player}(T(s,a)) =
\begin{cases}
p, & \text{if the last stone lands in } \operatorname{store}(p), \\
p, & \text{if the move captures stones}, \\
1-p, & \text{otherwise}.
\end{cases}
$$

For example, from the standard opening position with four stones in every pit,
player $0$ can choose local pit $2$. The four stones land in pits $3$, $4$,
$5$, and then player $0$'s store, so player $0$ gets another move.

The implementation records this in one line:

```python
next_player = mover if index == own_store or captured_stones else 1 - mover
```

This rule matters for search. Minimax and Monte Carlo Tree Search (MCTS) must look at
`state.current_player` after applying a move; they cannot simply assume that
turns alternate.

## Terminal States and Reward

The terminal predicate is:

$$
\operatorname{terminal}(s) =
\left[\sum_{i \in P_0} b_i = 0\right]
\lor
\left[\sum_{i \in P_1} b_i = 0\right].
$$

When the game ends, remaining pit stones are swept into stores by ownership, not
by who caused the ending. If player $0$ empties their side while player $1$
still has stones, those remaining stones go to player $1$'s store:

$$
\operatorname{store}_p
\leftarrow
\operatorname{store}_p + \sum_{i \in P_p} b_i
\qquad
\text{for each } p \in \{0,1\}.
$$

So the player who makes the final move does not automatically collect all
remaining stones. Each player collects the stones left on their own side.

The final reward for player $p$ is:

$$
z_p(s) =
\begin{cases}
1, & \operatorname{store}_p(s) > \operatorname{store}_{1-p}(s), \\
0, & \operatorname{store}_p(s) = \operatorname{store}_{1-p}(s), \\
-1, & \operatorname{store}_p(s) < \operatorname{store}_{1-p}(s).
\end{cases}
$$

Code:

```python
def reward_for_player(self, player: int) -> float:
    """Return +1, 0, or -1 for win, draw, or loss for the given player."""
    own = self.store_for(player)
    other = self.store_for(1 - player)
    if own > other:
        return 1.0
    if own < other:
        return -1.0
    return 0.0
```

## Practice

Run:

```bash
python scripts/play_cli.py --agent greedy
python -m pytest tests/test_game.py
```

Then inspect `tests/test_game.py` and explain which Kalah rule each test
protects.
