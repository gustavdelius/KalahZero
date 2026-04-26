from __future__ import annotations

from kalah_zero.game import GameState


ENCODING_VERSION = "fixed_count_v1"
PIT_STONE_SCALE = 18.0
STORE_STONE_SCALE = 72.0


def encode_features(state: GameState) -> list[float]:
    """Return a canonical current-player feature vector.

    The vector is:

    `[own pits, opponent pits reversed, own store, opponent store, bias]`

    Reversing the opponent pits lines up opposite pits in the same column. Pit
    and store counts are divided by fixed constants so the network sees actual
    stone counts while keeping inputs in a small numeric range.
    """

    player = state.current_player
    opponent = 1 - player
    own = [stones / PIT_STONE_SCALE for stones in state.pits_for(player)]
    other = [stones / PIT_STONE_SCALE for stones in reversed(state.pits_for(opponent))]
    stores = [
        state.store_for(player) / STORE_STONE_SCALE,
        state.store_for(opponent) / STORE_STONE_SCALE,
    ]
    return own + other + stores + [1.0]


def encode_state(state: GameState):
    import torch

    return torch.tensor(encode_features(state), dtype=torch.float32)


def input_size(pits: int = 6) -> int:
    return 2 * pits + 3
