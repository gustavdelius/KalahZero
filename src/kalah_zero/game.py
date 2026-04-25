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
        if pits <= 0:
            raise ValueError("pits must be positive")
        if stones < 0:
            raise ValueError("stones must be non-negative")
        board = [stones] * pits + [0] + [stones] * pits + [0]
        return cls(tuple(board), current_player=PLAYER_0, pits=pits)

    @property
    def store_0(self) -> int:
        return self.pits

    @property
    def store_1(self) -> int:
        return 2 * self.pits + 1

    @property
    def total_stones(self) -> int:
        return sum(self.board)

    def other_player(self) -> int:
        return 1 - self.current_player

    def store_index(self, player: int) -> int:
        self._validate_player(player)
        return self.store_0 if player == PLAYER_0 else self.store_1

    def pit_indices(self, player: int) -> range:
        self._validate_player(player)
        if player == PLAYER_0:
            return range(0, self.pits)
        return range(self.pits + 1, 2 * self.pits + 1)

    def pit_index(self, player: int, action: int) -> int:
        self._validate_player(player)
        if not 0 <= action < self.pits:
            raise ValueError(f"action must be in 0..{self.pits - 1}")
        return action if player == PLAYER_0 else self.pits + 1 + action

    def action_for_index(self, player: int, index: int) -> int:
        self._validate_player(player)
        if player == PLAYER_0:
            return index
        return index - (self.pits + 1)

    def pits_for(self, player: int) -> tuple[int, ...]:
        self._validate_player(player)
        if player == PLAYER_0:
            return self.board[: self.pits]
        return self.board[self.pits + 1 : 2 * self.pits + 1]

    def store_for(self, player: int) -> int:
        self._validate_player(player)
        return self.board[self.store_0 if player == PLAYER_0 else self.store_1]

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        if self.current_player == PLAYER_0:
            return [index for index in range(self.pits) if self.board[index] > 0]
        start = self.pits + 1
        return [index - start for index in range(start, start + self.pits) if self.board[index] > 0]

    def is_terminal(self) -> bool:
        return self._side_empty(PLAYER_0) or self._side_empty(PLAYER_1)

    def apply(self, action: int) -> GameState:
        if self.is_terminal():
            raise ValueError("cannot apply an action to a terminal state")

        mover = self.current_player
        if not 0 <= action < self.pits:
            raise ValueError(f"illegal action {action}")
        source = action if mover == PLAYER_0 else self.pits + 1 + action
        if self.board[source] <= 0:
            raise ValueError(f"illegal action {action}")

        board = list(self.board)
        stones = board[source]
        board[source] = 0

        own_store = self.store_0 if mover == PLAYER_0 else self.store_1
        opponent_store = self.store_1 if mover == PLAYER_0 else self.store_0
        index = source
        while stones:
            index = (index + 1) % len(board)
            if index == opponent_store:
                continue
            board[index] += 1
            stones -= 1

        captured_stones = False
        if self._is_own_pit(index, mover) and board[index] == 1:
            opposite = 2 * self.pits - index
            captured = board[opposite]
            if captured > 0:
                board[opposite] = 0
                board[index] = 0
                board[own_store] += captured + 1
                captured_stones = True

        next_player = mover if index == own_store or captured_stones else 1 - mover
        if self._side_empty_on_board(board, PLAYER_0) or self._side_empty_on_board(board, PLAYER_1):
            self._sweep_remaining(board)

        return GameState(tuple(board), current_player=next_player, pits=self.pits)

    def opposite_index(self, index: int) -> int:
        if index in (self.store_0, self.store_1):
            raise ValueError("stores do not have opposite pits")
        if not self._is_pit(index):
            raise ValueError(f"index {index} is outside the board")
        return 2 * self.pits - index

    def score_for_player(self, player: int) -> int:
        return self.store_for(player)

    def reward_for_player(self, player: int) -> float:
        self._validate_player(player)
        own = self.store_for(player)
        other = self.store_for(1 - player)
        if own > other:
            return 1.0
        if own < other:
            return -1.0
        return 0.0

    def normalized_store_margin(self, player: int) -> float:
        margin = self.store_for(player) - self.store_for(1 - player)
        return margin / max(1, self.total_stones)

    def render(self) -> str:
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
        return self._side_empty_on_board(self.board, player)

    def _side_empty_on_board(self, board: Sequence[int], player: int) -> bool:
        if player == PLAYER_0:
            for index in range(self.pits):
                if board[index] != 0:
                    return False
            return True
        start = self.pits + 1
        for index in range(start, start + self.pits):
            if board[index] != 0:
                return False
        return True

    def _sweep_remaining(self, board: list[int]) -> None:
        for index in range(self.pits):
            board[self.store_0] += board[index]
            board[index] = 0
        start = self.pits + 1
        for index in range(start, start + self.pits):
            board[self.store_1] += board[index]
            board[index] = 0

    def _is_pit(self, index: int) -> bool:
        return 0 <= index < len(self.board) and index not in (self.store_0, self.store_1)

    def _is_own_pit(self, index: int, player: int) -> bool:
        if player == PLAYER_0:
            return 0 <= index < self.pits
        return self.pits < index < self.store_1

    def _validate_player(self, player: int) -> None:
        if player not in (PLAYER_0, PLAYER_1):
            raise ValueError("player must be 0 or 1")
