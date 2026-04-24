#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import _path  # noqa: F401
from kalah_zero.mcts import UniformEvaluator
from kalah_zero.train import TrainConfig, self_play_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play samples with a uniform evaluator.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--output", default="self_play_samples.pkl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = TrainConfig(simulations=args.simulations, seed=args.seed)
    evaluator = UniformEvaluator()
    samples = []
    for game in range(args.games):
        game_samples = self_play_game(evaluator, config, rng)
        samples.extend(game_samples)
        print(f"game {game + 1}: {len(game_samples)} positions")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(samples, handle)
    print(f"wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()

