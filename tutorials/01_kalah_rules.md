# 01. Kalah Rules and State Representation

## Goal

By the end of this lesson, you should be able to explain exactly what a Kalah
state is, how an action changes it, and why an AlphaZero project starts with a
small, trustworthy game engine.

AlphaZero does not learn from pixels here. It learns from a formal game model.
For Kalah, that model is a tuple:

$$
s = (b, p),
$$

where $b$ is the board vector and $p \in \{0, 1\}$ is the player to move.

## Board Coordinates

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
indices $7,\ldots,12$. A move is represented as a local pit number
$a \in \{0,\ldots,5\}$ for the player to move.

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

The state is immutable because search algorithms reuse states constantly. If
`apply` mutated a board in place, one branch of the search tree could corrupt
another branch.

## Legal Actions

The legal action set is:

$$
\mathcal{A}(s) =
\{a \mid a \text{ is one of the current player's pits and } b_a > 0\}.
$$

In code, the definition is deliberately plain:

```python
def legal_actions(self) -> list[int]:
    if self.is_terminal():
        return []
    return [
        self.action_for_index(self.current_player, index)
        for index in self.pit_indices(self.current_player)
        if self.board[index] > 0
    ]
```

That code is worth reading slowly: it expresses the rule, not an optimization.
Teaching code should make the mathematical object easy to see.

## Transition Function

The transition function is:

$$
s' = T(s, a).
$$

For Kalah, $T$ does five things:

1. Pick up all stones from the selected pit.
2. Sow them counter-clockwise.
3. Skip the opponent's store.
4. Capture when the last stone lands in an empty own pit opposite stones.
5. Sweep remaining stones if either side becomes empty.

The public API is small:

```python
def apply(self, action: int) -> GameState:
    if self.is_terminal():
        raise ValueError("cannot apply an action to a terminal state")
    if action not in self.legal_actions():
        raise ValueError(f"illegal action {action}")
    ...
    return GameState(tuple(board), current_player=next_player, pits=self.pits)
```

The important design point is that all later algorithms see Kalah only through
this transition function. MCTS, minimax, and self-play do not need special
knowledge of sowing or capture.

## Terminal States and Reward

The terminal predicate is:

$$
\operatorname{terminal}(s) =
\left[\sum_{i \in P_0} b_i = 0\right]
\lor
\left[\sum_{i \in P_1} b_i = 0\right].
$$

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

