#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import _path  # noqa: F401
from kalah_zero.network import KalahNet, NeuralEvaluator
from kalah_zero.train import ReplayBuffer, TrainConfig, self_play_game, train_step


DEFAULT_OUTPUT = "checkpoints/kalah_zero.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small AlphaZero-style training loop.")
    parser.add_argument("--games", type=int, help="Total self-play games to complete.")
    parser.add_argument("--simulations", type=int, help="MCTS simulations per move.")
    parser.add_argument("--epochs", type=int, help="Training epochs after each self-play game.")
    parser.add_argument("--batch-size", type=int, help="Replay samples per training batch.")
    parser.add_argument("--output", help=f"Checkpoint path. Defaults to {DEFAULT_OUTPUT!r}.")
    parser.add_argument("--resume", help="Resume from an existing training checkpoint.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save every N completed games. Use 0 to save only at the end or on interrupt.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for a fresh run.")
    parser.add_argument(
        "--batched-mcts",
        dest="use_batched_mcts",
        action="store_true",
        default=None,
        help="Batch neural leaf evaluations during MCTS self-play.",
    )
    parser.add_argument(
        "--no-batched-mcts",
        dest="use_batched_mcts",
        action="store_false",
        help="Disable batched MCTS when resuming a checkpoint that used it.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Number of MCTS leaf positions to evaluate at once when using --batched-mcts.",
    )
    return parser


def config_from_args(args: argparse.Namespace, saved_config: TrainConfig | None = None) -> TrainConfig:
    base = saved_config or TrainConfig()
    return TrainConfig(
        pits=base.pits,
        stones=base.stones,
        simulations=args.simulations if args.simulations is not None else base.simulations,
        games_per_iteration=args.games if args.games is not None else base.games_per_iteration,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        learning_rate=base.learning_rate,
        weight_decay=base.weight_decay,
        replay_capacity=base.replay_capacity,
        temperature_moves=base.temperature_moves,
        seed=args.seed if args.seed is not None else base.seed,
        use_batched_mcts=(
            args.use_batched_mcts
            if args.use_batched_mcts is not None
            else base.use_batched_mcts
        ),
        eval_batch_size=(
            args.eval_batch_size
            if args.eval_batch_size is not None
            else base.eval_batch_size
        ),
    )


def make_mcts_factory(config: TrainConfig, rng: random.Random):
    if not config.use_batched_mcts:
        return None

    from kalah_zero.batched_mcts import BatchedMCTS

    return lambda: BatchedMCTS(
        simulations=config.simulations,
        dirichlet_alpha=0.3,
        rng=rng,
        batch_size=config.eval_batch_size,
    )


def checkpoint_path(args: argparse.Namespace) -> Path:
    return Path(args.output or args.resume or DEFAULT_OUTPUT)


def save_training_checkpoint(
    path: Path,
    model: KalahNet,
    optimizer,
    buffer: ReplayBuffer,
    rng: random.Random,
    config: TrainConfig,
    completed_games: int,
    reason: str,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "pits": model.pits,
        "config": asdict(config),
        "completed_games": completed_games,
        "replay_capacity": buffer.capacity,
        "replay_samples": buffer.samples,
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    print(f"saved checkpoint to {path} ({reason}, completed_games={completed_games})")


def load_training_checkpoint(path: Path):
    import torch

    from kalah_zero.network import load_checkpoint

    model, payload = load_checkpoint(str(path))
    saved_config = TrainConfig(**payload.get("config", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=saved_config.learning_rate,
        weight_decay=saved_config.weight_decay,
    )
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    completed_games = int(payload.get("completed_games", payload.get("step", 0)))
    rng = random.Random(saved_config.seed)
    if "rng_state" in payload:
        rng.setstate(payload["rng_state"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    buffer = ReplayBuffer(
        capacity=int(payload.get("replay_capacity", saved_config.replay_capacity)),
        rng=rng,
        samples=list(payload.get("replay_samples", [])),
    )
    return model, optimizer, buffer, rng, saved_config, completed_games


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import torch

    output = checkpoint_path(args)
    if args.resume:
        model, optimizer, buffer, rng, saved_config, completed_games = load_training_checkpoint(Path(args.resume))
        config = config_from_args(args, saved_config)
        print(
            f"resumed {args.resume}: completed_games={completed_games}, "
            f"target_games={config.games_per_iteration}, buffer={len(buffer)}, "
            f"batched_mcts={config.use_batched_mcts}"
        )
    else:
        config = config_from_args(args)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        rng = random.Random(config.seed)
        model = KalahNet(pits=config.pits)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        buffer = ReplayBuffer(config.replay_capacity, rng=rng)
        completed_games = 0

    if config.games_per_iteration < completed_games:
        raise ValueError(
            f"target games ({config.games_per_iteration}) is less than completed games ({completed_games})"
        )

    try:
        for game_index in range(completed_games, config.games_per_iteration):
            evaluator = NeuralEvaluator(model)
            samples = self_play_game(
                evaluator,
                config,
                rng,
                mcts_factory=make_mcts_factory(config, rng),
            )
            buffer.add_many(samples)
            completed_games = game_index + 1
            print(f"game {completed_games}: collected {len(samples)} positions, buffer={len(buffer)}")

            for epoch in range(config.epochs):
                metrics = train_step(model, optimizer, buffer.sample(config.batch_size))
                print(
                    f"  epoch {epoch + 1}: loss={metrics['loss']:.4f} "
                    f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}"
                )

            if args.checkpoint_every > 0 and completed_games % args.checkpoint_every == 0:
                save_training_checkpoint(
                    output,
                    model,
                    optimizer,
                    buffer,
                    rng,
                    config,
                    completed_games,
                    reason="periodic",
                )
    except KeyboardInterrupt:
        print("\ninterrupted; saving before exit...")
        save_training_checkpoint(
            output,
            model,
            optimizer,
            buffer,
            rng,
            config,
            completed_games,
            reason="interrupt",
        )
        raise SystemExit(130)

    save_training_checkpoint(
        output,
        model,
        optimizer,
        buffer,
        rng,
        config,
        completed_games,
        reason="final",
    )


if __name__ == "__main__":
    main()
