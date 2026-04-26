from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PLAYER_0 = 0
PLAYER_1 = 1


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable Kalah state.

    Board layout for `pits = 6`:

    - indices 0..5 are player 0 pits
    - index 6 is player 0 store
    - indices 7..12 are player 1 pits
    - index 13 is player 1 store

    Actions are always local pit numbers 0..pits-1 for the player to move.
    """

    board: tuple[int, ...]
    current_player: int
    pits: int = 6

    @classmethod
    def new_game(cls, pits: int = 6, stones: int = 4) -> GameState:
        """Create the standard opening position with every pit filled equally."""
        if pits <= 0:
            raise ValueError("pits must be positive")
        if stones < 0:
            raise ValueError("stones must be non-negative")
        board = [stones] * pits + [0] + [stones] * pits + [0]
        return cls(tuple(board), current_player=PLAYER_0, pits=pits)

    @property
    def store_0(self) -> int:
        """Board index of player 0's store."""
        return self.pits

    @property
    def store_1(self) -> int:
        """Board index of player 1's store."""
        return 2 * self.pits + 1

    @property
    def total_stones(self) -> int:
        """Total number of stones on the board (constant until terminal sweep)."""
        return sum(self.board)

    def other_player(self) -> int:
        """Return the player who is not the current player."""
        return 1 - self.current_player

    def store_index(self, player: int) -> int:
        """Return the board index of the given player's store."""
        self._validate_player(player)
        return self.store_0 if player == PLAYER_0 else self.store_1

    def pit_indices(self, player: int) -> range:
        """Return the range of board indices covering the given player's pits."""
        self._validate_player(player)
        if player == PLAYER_0:
            return range(0, self.pits)
        # +1 skips over player 0's store at index `pits`
        return range(self.pits + 1, 2 * self.pits + 1)

    def pit_index(self, player: int, action: int) -> int:
        """Convert a local action number (0..pits-1) to a board index."""
        self._validate_player(player)
        if not 0 <= action < self.pits:
            raise ValueError(f"action must be in 0..{self.pits - 1}")
        # Player 1's pits start at pits+1 (skipping player 0's store at index pits)
        return action if player == PLAYER_0 else self.pits + 1 + action

    def action_for_index(self, player: int, index: int) -> int:
        """Convert a board index back to a local action number (inverse of pit_index)."""
        self._validate_player(player)
        if player == PLAYER_0:
            return index
        return index - (self.pits + 1)

    def pits_for(self, player: int) -> tuple[int, ...]:
        """Return the stone counts in the given player's pits, in board order."""
        return tuple(self.board[i] for i in self.pit_indices(player))

    def store_for(self, player: int) -> int:
        """Return the number of stones in the given player's store."""
        return self.board[self.store_index(player)]

    def legal_actions(self) -> list[int]:
        """Return the list of local action numbers the current player may choose."""
        if self.is_terminal():
            return []
        return [
            self.action_for_index(self.current_player, index)
            for index in self.pit_indices(self.current_player)
            if self.board[index] > 0
        ]

    def is_terminal(self) -> bool:
        """Return True when at least one player's pits are all empty."""
        return self._side_empty(PLAYER_0) or self._side_empty(PLAYER_1)

    def apply(self, action: int) -> GameState:
        """Return the state that results from the current player choosing `action`."""
        if self.is_terminal():
            raise ValueError("cannot apply an action to a terminal state")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")

        board = list(self.board)
        mover = self.current_player
        source = self.pit_index(mover, action)
        stones = board[source]
        board[source] = 0

        index = source
        while stones:
            index = (index + 1) % len(board)
            if index == self.store_index(1 - mover):  # skip opponent's store when sowing
                continue
            board[index] += 1
            stones -= 1

        own_store = self.store_index(mover)
        captured_stones = False
        # board[index] == 1 means the pit held 0 stones before this move (we just placed the 1st)
        if self._is_own_pit(index, mover) and board[index] == 1:
            opposite = self.opposite_index(index)
            captured = board[opposite]
            if captured > 0:
                board[opposite] = 0
                board[index] = 0
                board[own_store] += captured + 1  # opponent's stones + the landing stone
                captured_stones = True

        # Extra turn if last stone landed in own store, or if the move made a capture
        next_player = mover if index == own_store or captured_stones else 1 - mover
        if self._side_empty_on_board(board, PLAYER_0) or self._side_empty_on_board(board, PLAYER_1):
            self._sweep_remaining(board)

        return GameState(tuple(board), current_player=next_player, pits=self.pits)

    def opposite_index(self, index: int) -> int:
        """Return the board index of the pit directly opposite `index`.

        Works because p0_index + p1_index = 2*pits for every facing pair
        (e.g. 0+12=12, 5+7=12 when pits=6).
        """
        if index in (self.store_0, self.store_1):
            raise ValueError("stores do not have opposite pits")
        if not self._is_pit(index):
            raise ValueError(f"index {index} is outside the board")
        return 2 * self.pits - index

    def score_for_player(self, player: int) -> int:
        """Return the current player's score (number of stones in their store)."""
        return self.store_for(player)

    def reward_for_player(self, player: int) -> float:
        """Return +1, 0, or -1 for win, draw, or loss for the given player."""
        self._validate_player(player)
        own = self.store_for(player)
        other = self.store_for(1 - player)
        if own > other:
            return 1.0
        if own < other:
            return -1.0
        return 0.0

    def normalized_store_margin(self, player: int) -> float:
        """Return the store margin for `player` divided by total stones on the board.

        Used as a fast heuristic score: positive means the player is ahead,
        negative means behind, and the scale is always in [-1, 1].
        """
        margin = self.store_for(player) - self.store_for(1 - player)
        return margin / max(1, self.total_stones)

    def render(self) -> str:
        """Return a human-readable string showing the board from player 0's perspective."""
        top = " ".join(f"{n:2d}" for n in reversed(self.pits_for(PLAYER_1)))
        bottom = " ".join(f"{n:2d}" for n in self.pits_for(PLAYER_0))
        return (
            f"      {top}\n"
            f"P1 {self.store_for(PLAYER_1):2d}"
            f"{' ' * (len(top) + 2)}"
            f"{self.store_for(PLAYER_0):2d} P0\n"
            f"      {bottom}\n"
            f"to move: P{self.current_player}"
        )

    def _side_empty(self, player: int) -> bool:
        """Return True if all of the given player's pits are empty."""
        return self._side_empty_on_board(self.board, player)

    def _side_empty_on_board(self, board: Sequence[int], player: int) -> bool:
        """Return True if all of the given player's pits are empty in `board`."""
        return all(board[i] == 0 for i in self.pit_indices(player))

    def _sweep_remaining(self, board: list[int]) -> None:
        """Move all remaining pit stones into their owners' stores (end-of-game sweep)."""
        for player in (PLAYER_0, PLAYER_1):
            store = self.store_index(player)
            for index in self.pit_indices(player):
                board[store] += board[index]
                board[index] = 0

    def _is_pit(self, index: int) -> bool:
        """Return True if `index` is a valid pit (not a store, not out of range)."""
        return 0 <= index < len(self.board) and index not in (self.store_0, self.store_1)

    def _is_own_pit(self, index: int, player: int) -> bool:
        """Return True if `index` is one of `player`'s own pits."""
        return index in self.pit_indices(player)

    def _validate_player(self, player: int) -> None:
        """Raise ValueError if `player` is not 0 or 1."""
        if player not in (PLAYER_0, PLAYER_1):
            raise ValueError("player must be 0 or 1")
