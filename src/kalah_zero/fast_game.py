from __future__ import annotations

from dataclasses import dataclass

from kalah_zero import _fast_game
from kalah_zero.game import PLAYER_0, PLAYER_1


@dataclass(frozen=True, slots=True)
class FastGameState:
    """A C++-accelerated Kalah state with the same public API as `GameState`."""

    board: tuple[int, ...]
    current_player: int
    pits: int = 6

    @classmethod
    def new_game(cls, pits: int = 6, stones: int = 4) -> FastGameState:
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
        return _fast_game.pits_for(self.board, self.pits, player)

    def store_for(self, player: int) -> int:
        return _fast_game.store_for(self.board, self.pits, player)

    def legal_actions(self) -> list[int]:
        return _fast_game.legal_actions(self.board, self.current_player, self.pits)

    def is_terminal(self) -> bool:
        return _fast_game.is_terminal(self.board, self.pits)

    def apply(self, action: int) -> FastGameState:
        board, next_player = _fast_game.apply(self.board, self.current_player, self.pits, action)
        return FastGameState(board, current_player=next_player, pits=self.pits)

    def opposite_index(self, index: int) -> int:
        if index in (self.store_0, self.store_1):
            raise ValueError("stores do not have opposite pits")
        if not self._is_pit(index):
            raise ValueError(f"index {index} is outside the board")
        return 2 * self.pits - index

    def score_for_player(self, player: int) -> int:
        return self.store_for(player)

    def reward_for_player(self, player: int) -> float:
        return _fast_game.reward_for_player(self.board, self.pits, player)

    def normalized_store_margin(self, player: int) -> float:
        return _fast_game.normalized_store_margin(self.board, self.pits, player)

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

    def _is_pit(self, index: int) -> bool:
        return 0 <= index < len(self.board) and index not in (self.store_0, self.store_1)

    def _validate_player(self, player: int) -> None:
        if player not in (PLAYER_0, PLAYER_1):
            raise ValueError("player must be 0 or 1")
