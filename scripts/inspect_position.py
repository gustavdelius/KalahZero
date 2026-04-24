#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _path  # noqa: F401
from kalah_zero.game import GameState
from kalah_zero.mcts import MCTS, UniformEvaluator


def parse_state(raw: str | None, player: int) -> GameState:
    if raw is None:
        return GameState.new_game()
    board = tuple(int(part.strip()) for part in raw.split(","))
    pits = (len(board) - 2) // 2
    if len(board) != 2 * pits + 2:
        raise ValueError("board must contain 2*pits+2 comma-separated integers")
    return GameState(board=board, current_player=player, pits=pits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MCTS recommendations for a Kalah position.")
    parser.add_argument("--board", help="Comma-separated board, including stores.")
    parser.add_argument("--player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--tree-depth", type=int, default=1)
    args = parser.parse_args()

    state = parse_state(args.board, args.player)
    mcts = MCTS(simulations=args.simulations)
    result = mcts.search(state, UniformEvaluator())
    print(state.render())
    print()
    for action, (visits, prob) in enumerate(zip(result.visits, result.policy)):
        if visits:
            print(f"pit {action}: visits={visits:4d} prob={prob:.3f}")
    print()
    print(mcts.dump_tree(result.root, max_depth=args.tree_depth))


if __name__ == "__main__":
    main()

