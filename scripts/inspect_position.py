#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import _path  # noqa: F401
from kalah_zero.game import GameState
from kalah_zero.mcts import MCTS, SearchNode, UniformEvaluator


def build_evaluator(checkpoint: str | None):
    """Return a NeuralEvaluator for the given checkpoint, or a UniformEvaluator if none is given."""
    if checkpoint is None:
        return UniformEvaluator()
    from kalah_zero.network import NeuralEvaluator, load_checkpoint
    model, _ = load_checkpoint(checkpoint)
    return NeuralEvaluator(model)


def path_actions(path: list[SearchNode]) -> list[tuple[int, int]]:
    """Return (action, player) pairs from root to leaf by inspecting each parent's children dict."""
    actions = []
    for parent, child in zip(path, path[1:]):
        for action, node in parent.children.items():
            if node is child:
                actions.append((action, parent.state.current_player))
                break
    return actions


def parse_state(raw: str | None, player: int) -> GameState:
    if raw is None:
        return GameState.new_game()
    board = tuple(int(part.strip()) for part in raw.split(","))
    pits = (len(board) - 2) // 2
    if len(board) != 2 * pits + 2:
        raise ValueError("board must contain 2*pits+2 comma-separated integers")
    return GameState(board=board, current_player=player, pits=pits)


def render_tree(state: GameState, root: SearchNode, mcts: MCTS, sim: int, total: int, depth: int) -> str:
    """Return a string showing the board, per-pit stats, and the search tree for one snapshot."""
    lines = [state.render(), f"\nSimulation {sim} / {total}"]
    for action, child in sorted(root.children.items()):
        lines.append(
            f"  pit {action}: visits={child.visit_count:4d}  Q={child.mean_value:+.3f}  prior={child.prior:.3f}"
        )
    lines.append("")
    lines.append(mcts.dump_tree(root, max_depth=depth))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MCTS recommendations for a Kalah position.")
    parser.add_argument("--board", help="Comma-separated board, including stores.")
    parser.add_argument("--player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--checkpoint", help="Path to a trained network checkpoint (.pt file).")
    parser.add_argument("--tree-depth", type=int, default=1)
    parser.add_argument("--watch", action="store_true",
                        help="Animate the tree as simulations run.")
    parser.add_argument("--watch-every", type=int, default=1, metavar="N",
                        help="Redraw every N simulations when --watch is set (default 1).")
    parser.add_argument("--delay", type=float, default=0.05, metavar="SECS",
                        help="Pause between redraws in seconds when --watch is set (default 0.05).")
    parser.add_argument("--trace", action="store_true",
                        help="Print one line per simulation: the action path and the leaf's prior.")
    args = parser.parse_args()

    state = parse_state(args.board, args.player)
    mcts = MCTS(simulations=args.simulations)
    evaluator = build_evaluator(args.checkpoint)

    callback = None
    if args.watch:
        every = args.watch_every
        delay = args.delay
        depth = args.tree_depth
        def callback(root: SearchNode, sim: int, path: list[SearchNode]) -> None:
            """Clear the terminal and redraw the tree snapshot."""
            if sim % every != 0:
                return
            # ANSI escape: clear screen and move cursor to top-left.
            print("\033[2J\033[H", end="", flush=True)
            print(render_tree(state, root, mcts, sim, args.simulations, depth), flush=True)
            if delay:
                time.sleep(delay)
    elif args.trace:
        def callback(root: SearchNode, sim: int, path: list[SearchNode]) -> None:
            """Print the action path to the selected leaf and that leaf's prior."""
            actions = path_actions(path)
            action_str = " → ".join(f"P{p}:{a}" for a, p in actions) if actions else "(root)"
            print(f"sim {sim:4d}: {action_str:30s}  prior={path[-1].prior:.3f}")

    result = mcts.search(state, evaluator, callback=callback)

    if args.watch:
        print("\033[2J\033[H", end="", flush=True)
    if args.trace:
        print()
    print(state.render())
    print()
    for action, (visits, prob) in enumerate(zip(result.visits, result.policy)):
        if visits:
            print(f"pit {action}: visits={visits:4d} prob={prob:.3f}")
    print()
    print(mcts.dump_tree(result.root, max_depth=args.tree_depth))


if __name__ == "__main__":
    main()

