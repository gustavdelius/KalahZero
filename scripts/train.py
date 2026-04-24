#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import _path  # noqa: F401
from kalah_zero.network import KalahNet, NeuralEvaluator, save_checkpoint
from kalah_zero.train import ReplayBuffer, TrainConfig, self_play_game, train_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small AlphaZero-style training loop.")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="checkpoints/kalah_zero.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    config = TrainConfig(
        simulations=args.simulations,
        games_per_iteration=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = KalahNet(pits=config.pits)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    buffer = ReplayBuffer(config.replay_capacity, rng=rng)

    for game in range(config.games_per_iteration):
        evaluator = NeuralEvaluator(model)
        samples = self_play_game(evaluator, config, rng)
        buffer.add_many(samples)
        print(f"game {game + 1}: collected {len(samples)} positions, buffer={len(buffer)}")

        for epoch in range(config.epochs):
            metrics = train_step(model, optimizer, buffer.sample(config.batch_size))
            print(
                f"  epoch {epoch + 1}: loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(output), model, optimizer, step=config.games_per_iteration)
    print(f"saved checkpoint to {output}")


if __name__ == "__main__":
    main()

