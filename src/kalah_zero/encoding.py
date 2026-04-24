from __future__ import annotations

from kalah_zero.game import GameState


def encode_features(state: GameState) -> list[float]:
    """Return a canonical current-player feature vector.

    The vector is:

    `[own pits, opponent pits reversed, own store, opponent store, bias]`

    Reversing the opponent pits lines up opposite pits in the same column.
    """

    player = state.current_player
    opponent = 1 - player
    scale = float(max(1, state.total_stones))
    own = [stones / scale for stones in state.pits_for(player)]
    other = [stones / scale for stones in reversed(state.pits_for(opponent))]
    stores = [state.store_for(player) / scale, state.store_for(opponent) / scale]
    return own + other + stores + [1.0]


def encode_state(state: GameState):
    import torch

    return torch.tensor(encode_features(state), dtype=torch.float32)


def input_size(pits: int = 6) -> int:
    return 2 * pits + 3

