#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import _path  # noqa: F401
from kalah_zero.agents import GreedyAgent, MCTSAgent, MinimaxAgent, RandomAgent
from kalah_zero.evaluate import arena
from kalah_zero.mcts import MCTS, UniformEvaluator


def build_agent(name: str, seed: int, checkpoint: str | None = None, simulations: int = 100):
    if checkpoint is not None:
        from kalah_zero.network import NeuralEvaluator, load_checkpoint

        model, _ = load_checkpoint(checkpoint)
        return MCTSAgent(MCTS(simulations=simulations), NeuralEvaluator(model), temperature=0.0)
    if name == "random":
        return RandomAgent(random.Random(seed))
    if name == "greedy":
        return GreedyAgent()
    if name == "minimax":
        return MinimaxAgent(depth=6)
    if name == "mcts":
        return MCTSAgent(MCTS(simulations=simulations), UniformEvaluator(), temperature=0.0)
    raise ValueError(f"unknown agent {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two Kalah agents.")
    parser.add_argument("--agent-a", choices=["random", "greedy", "minimax", "mcts"], default="greedy")
    parser.add_argument("--agent-b", choices=["random", "greedy", "minimax", "mcts"], default="random")
    parser.add_argument("--checkpoint-a", help="Use a neural checkpoint for agent A.")
    parser.add_argument("--checkpoint-b", help="Use a neural checkpoint for agent B.")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = arena(
        build_agent(args.agent_a, args.seed, args.checkpoint_a, args.simulations),
        build_agent(args.agent_b, args.seed + 1, args.checkpoint_b, args.simulations),
        games=args.games,
        seed=args.seed,
    )
    label_a = args.checkpoint_a or args.agent_a
    label_b = args.checkpoint_b or args.agent_b
    print(
        f"{label_a} vs {label_b}: "
        f"{result.wins_0}-{result.wins_1}-{result.draws} "
        f"(win rate {result.win_rate_0:.3f})"
    )


if __name__ == "__main__":
    main()
