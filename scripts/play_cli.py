#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import _path  # noqa: F401
from kalah_zero.agents import GreedyAgent, MinimaxAgent, RandomAgent
from kalah_zero.game import GameState
from kalah_zero.mcts import MCTS, UniformEvaluator
from kalah_zero.agents import MCTSAgent


def build_agent(name: str, seed: int):
    if name == "random":
        return RandomAgent(random.Random(seed))
    if name == "greedy":
        return GreedyAgent()
    if name == "minimax":
        return MinimaxAgent(depth=6)
    if name == "mcts":
        return MCTSAgent(MCTS(simulations=100), UniformEvaluator(), temperature=0.0)
    raise ValueError(f"unknown agent {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Kalah against a baseline agent.")
    parser.add_argument("--agent", choices=["random", "greedy", "minimax", "mcts"], default="greedy")
    parser.add_argument("--human-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    state = GameState.new_game()
    agent = build_agent(args.agent, args.seed)

    while not state.is_terminal():
        print()
        print(state.render())
        legal = state.legal_actions()
        if state.current_player == args.human_player:
            action = None
            while action not in legal:
                raw = input(f"Choose pit {legal}: ")
                try:
                    action = int(raw)
                except ValueError:
                    action = None
            state = state.apply(action)
        else:
            action = agent.select_action(state)
            print(f"Agent chooses pit {action}")
            state = state.apply(action)

    print()
    print(state.render())
    print(f"Final score: P0={state.score_for_player(0)} P1={state.score_for_player(1)}")


if __name__ == "__main__":
    main()

