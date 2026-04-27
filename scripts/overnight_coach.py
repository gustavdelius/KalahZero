#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import _path  # noqa: F401


RESULT_RE = re.compile(
    r":\s+(?P<wins>\d+)-(?P<losses>\d+)-(?P<draws>\d+)\s+"
    r"\(win rate\s+(?P<win_rate>[0-9.]+),"
)


@dataclass(frozen=True, slots=True)
class EvalSpec:
    opponent: str
    stones: int
    simulations: int
    games: int
    noise_prob: float | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    block: int
    checkpoint: str
    opponent: str
    stones: int
    simulations: int
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    score_rate: float
    noise_prob: float | None
    timestamp: str
    command: str


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_stone_weight_text(text: str | None, stones: list[int]) -> dict[int, float]:
    if text is None:
        return {stone: 1.0 for stone in stones}
    weights: dict[int, float] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("stone weights must look like '4:1,5:1,6:2'")
        raw_stones, raw_weight = item.split(":", 1)
        stone = int(raw_stones)
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError("stone weights must be positive")
        weights[stone] = weight
    for stone in stones:
        weights.setdefault(stone, 1.0)
    return {stone: weights[stone] for stone in stones}


def format_stone_weights(weights: dict[int, float]) -> str:
    return ",".join(f"{stone}:{weights[stone]:.3g}" for stone in sorted(weights))


def parse_result_line(text: str) -> tuple[int, int, int, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = RESULT_RE.search(line)
        if match:
            return (
                int(match.group("wins")),
                int(match.group("losses")),
                int(match.group("draws")),
                float(match.group("win_rate")),
            )
    raise ValueError(f"could not parse evaluation result from output:\n{text}")


def score_rate(wins: int, draws: int, games: int) -> float:
    return (wins + 0.5 * draws) / max(1, games)


def read_completed_games(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("completed_games", payload.get("step", 0)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run overnight train/evaluate cycles and keep the best low-simulation checkpoint."
    )
    parser.add_argument("--output-dir", default="runs/overnight_coach")
    parser.add_argument("--start-checkpoint", help="Optional checkpoint to continue from.")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-games", type=int, default=500)
    parser.add_argument("--checkpoint-prefix", default="fixed_scale")
    parser.add_argument("--train-simulations", type=int, default=250)
    parser.add_argument(
        "--train-simulations-schedule",
        help="Comma-separated per-block training simulations. The last value repeats.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--epochs-schedule",
        help="Comma-separated per-block epoch counts. The last value repeats.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--model-type", choices=["mlp", "residual"], default="residual")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-simulations", default="25,50,100,150")
    parser.add_argument("--stones", default="4,5,6")
    parser.add_argument(
        "--stone-weights",
        help="Weighted training stone distribution passed to train.py, e.g. '4:1,5:1,6:2'.",
    )
    parser.add_argument(
        "--balance-stone-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each block, adjust --stone-weights so weaker minimax stone "
            "counts are sampled more often and stronger ones less often."
        ),
    )
    parser.add_argument(
        "--stone-weight-balance-strength",
        type=float,
        default=1.0,
        help="Exponent applied to minimax score-ratio targets when balancing stone weights.",
    )
    parser.add_argument(
        "--stone-weight-smoothing",
        type=float,
        default=0.25,
        help=(
            "How much of the previous normalized stone weights to retain when "
            "balancing. Use 0 for purely score-based weights."
        ),
    )
    parser.add_argument(
        "--stone-weight-min",
        type=float,
        default=0.25,
        help="Minimum generated weight for any stone count when balancing stone weights.",
    )
    parser.add_argument(
        "--stone-weight-max",
        type=float,
        default=4.0,
        help="Maximum generated weight for any stone count when balancing stone weights.",
    )
    parser.add_argument("--opening-plies-min", type=int, default=0)
    parser.add_argument("--opening-plies-max", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--batched-mcts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-game", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum", choices=["mixed", "adaptive"], default="adaptive")
    parser.add_argument(
        "--focus-threshold",
        type=float,
        default=0.50,
        help="Adaptive mode focuses the weakest stone count while its low-sim score is below this.",
    )
    parser.add_argument(
        "--focus-margin",
        type=float,
        default=0.08,
        help="Adaptive mode focuses only if the weakest stone count trails the mean by this much.",
    )
    parser.add_argument("--include-noisy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noisy-simulations", default="50")
    parser.add_argument("--noise-probs", default="0.10")
    parser.add_argument("--low-sim-weight", type=float, default=3.0)
    parser.add_argument("--mid-sim-weight", type=float, default=2.0)
    parser.add_argument("--high-sim-weight", type=float, default=1.0)
    parser.add_argument("--noisy-weight", type=float, default=0.5)
    parser.add_argument("--parity-penalty", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def train_stone_args(
    stones: list[int],
    focus_stone: int | None,
    stone_weights: str | None = None,
) -> list[str]:
    if focus_stone is not None:
        return ["--stones", str(focus_stone)]
    if stone_weights is not None:
        return ["--stone-weights", stone_weights]
    return ["--stones-min", str(min(stones)), "--stones-max", str(max(stones))]


def scheduled_value(schedule_text: str | None, fallback: int, block: int) -> int:
    if not schedule_text:
        return fallback
    values = parse_csv_ints(schedule_text)
    if not values:
        return fallback
    return values[min(block - 1, len(values) - 1)]


def build_train_command(
    args: argparse.Namespace,
    checkpoint: Path,
    previous_checkpoint: Path | None,
    target_games: int,
    focus_stone: int | None,
    stones: list[int],
    train_simulations: int | None = None,
    epochs: int | None = None,
    stone_weights: str | None = None,
) -> list[str]:
    train_simulations = args.train_simulations if train_simulations is None else train_simulations
    epochs = args.epochs if epochs is None else epochs
    command = [
        sys.executable,
        "scripts/train.py",
        "--games",
        str(target_games),
        "--output",
        str(checkpoint),
        "--simulations",
        str(train_simulations),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--replay-capacity",
        str(args.replay_capacity),
        "--opening-plies-min",
        str(args.opening_plies_min),
        "--opening-plies-max",
        str(args.opening_plies_max),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    command.extend(train_stone_args(stones, focus_stone, stone_weights or args.stone_weights))
    if previous_checkpoint is None:
        command.extend([
            "--model-type",
            args.model_type,
            "--hidden-size",
            str(args.hidden_size),
            "--residual-blocks",
            str(args.residual_blocks),
            "--seed",
            str(args.seed),
        ])
    else:
        command.extend(["--resume", str(previous_checkpoint)])
    command.append("--batched-mcts" if args.batched_mcts else "--no-batched-mcts")
    command.append("--fast-game" if args.fast_game else "--no-fast-game")
    return command


def build_eval_specs(args: argparse.Namespace, stones: list[int]) -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for simulations in parse_csv_ints(args.eval_simulations):
        for stone_count in stones:
            specs.append(
                EvalSpec(
                    opponent="minimax",
                    stones=stone_count,
                    simulations=simulations,
                    games=args.eval_games,
                )
            )
    if args.include_noisy:
        for simulations in parse_csv_ints(args.noisy_simulations):
            for noise_prob in parse_csv_floats(args.noise_probs):
                for stone_count in stones:
                    specs.append(
                        EvalSpec(
                            opponent="noisy-minimax",
                            stones=stone_count,
                            simulations=simulations,
                            games=args.eval_games,
                            noise_prob=noise_prob,
                        )
                    )
    return specs


def build_eval_command(
    args: argparse.Namespace,
    checkpoint: Path,
    spec: EvalSpec,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint-a",
        str(checkpoint),
        "--agent-b",
        spec.opponent,
        "--games",
        str(spec.games),
        "--stones",
        str(spec.stones),
        "--simulations",
        str(spec.simulations),
        "--opening-plies-min",
        str(args.opening_plies_min),
        "--opening-plies-max",
        str(args.opening_plies_max),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--seed",
        str(seed),
    ]
    if args.batched_mcts:
        command.append("--batched-mcts")
    if args.fast_game:
        command.append("--fast-game")
    if spec.noise_prob is not None:
        command.extend(["--noise-prob", str(spec.noise_prob)])
    return command


def run_evaluation(
    args: argparse.Namespace,
    block: int,
    checkpoint: Path,
    spec: EvalSpec,
    seed: int,
) -> EvalResult:
    command = build_eval_command(args, checkpoint, spec, seed)
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    output = completed.stdout + "\n" + completed.stderr
    wins, losses, draws, win_rate = parse_result_line(output)
    result = EvalResult(
        block=block,
        checkpoint=str(checkpoint),
        opponent=spec.opponent,
        stones=spec.stones,
        simulations=spec.simulations,
        games=spec.games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        score_rate=score_rate(wins, draws, spec.games),
        noise_prob=spec.noise_prob,
        timestamp=now_iso(),
        command=command_text(command),
    )
    print(
        f"  eval {spec.opponent} stones={spec.stones} sims={spec.simulations}: "
        f"{wins}-{losses}-{draws} score={result.score_rate:.3f}"
    )
    return result


def simulation_weight(args: argparse.Namespace, simulations: int, all_simulations: list[int]) -> float:
    ordered = sorted(all_simulations)
    if simulations == ordered[0]:
        return args.low_sim_weight
    if simulations == ordered[-1]:
        return args.high_sim_weight
    return args.mid_sim_weight


def checkpoint_score(args: argparse.Namespace, results: list[EvalResult]) -> float:
    eval_sims = parse_csv_ints(args.eval_simulations)
    score = 0.0
    total_weight = 0.0
    deterministic = [r for r in results if r.opponent == "minimax"]
    for result in deterministic:
        weight = simulation_weight(args, result.simulations, eval_sims)
        score += weight * result.score_rate
        total_weight += weight
    noisy = [r for r in results if r.opponent == "noisy-minimax"]
    for result in noisy:
        score += args.noisy_weight * result.score_rate
        total_weight += args.noisy_weight
    normalized = score / max(1e-9, total_weight)
    low_sim = min(eval_sims)
    low_results = [r for r in deterministic if r.simulations == low_sim]
    penalty = sum(max(0.0, 0.50 - r.score_rate) for r in low_results)
    return normalized - args.parity_penalty * penalty / max(1, len(low_results))


def minimax_score_by_stone(
    args: argparse.Namespace,
    stones: list[int],
    results: list[EvalResult],
) -> dict[int, float]:
    eval_sims = parse_csv_ints(args.eval_simulations)
    totals = {stone: 0.0 for stone in stones}
    weights = {stone: 0.0 for stone in stones}
    for result in results:
        if result.opponent != "minimax" or result.stones not in totals:
            continue
        weight = simulation_weight(args, result.simulations, eval_sims)
        totals[result.stones] += weight * result.score_rate
        weights[result.stones] += weight
    return {
        stone: totals[stone] / weights[stone]
        for stone in stones
        if weights[stone] > 0
    }


def balanced_stone_weights(
    args: argparse.Namespace,
    stones: list[int],
    results: list[EvalResult],
    current_weights_text: str | None,
) -> str:
    current_weights = parse_stone_weight_text(current_weights_text, stones)
    scores = minimax_score_by_stone(args, stones, results)
    if not scores:
        return format_stone_weights(current_weights)
    target = sum(scores.values()) / len(scores)
    if target <= 0:
        return format_stone_weights(current_weights)
    target_weights: dict[int, float] = {}
    for stone in stones:
        score = max(scores.get(stone, target), 1e-6)
        target_weights[stone] = (target / score) ** args.stone_weight_balance_strength
    normalized_targets = normalize_weights(target_weights)
    normalized_current = normalize_weights(current_weights)
    smoothing = min(1.0, max(0.0, args.stone_weight_smoothing))
    blended = {
        stone: smoothing * normalized_current[stone] + (1.0 - smoothing) * normalized_targets[stone]
        for stone in stones
    }
    normalized = normalize_weights(blended)
    clamped = clamp_weights(normalized, args.stone_weight_min, args.stone_weight_max)
    return format_stone_weights(normalize_weights(clamped))


def normalize_weights(weights: dict[int, float]) -> dict[int, float]:
    mean_weight = sum(weights.values()) / max(1, len(weights))
    if mean_weight <= 0:
        return {stone: 1.0 for stone in weights}
    return {stone: weight / mean_weight for stone, weight in weights.items()}


def clamp_weights(weights: dict[int, float], minimum: float, maximum: float) -> dict[int, float]:
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    return {
        stone: min(maximum, max(minimum, weight))
        for stone, weight in weights.items()
    }


def choose_next_focus(
    args: argparse.Namespace,
    stones: list[int],
    results: list[EvalResult],
) -> int | None:
    if args.curriculum == "mixed":
        return None
    low_sim = min(parse_csv_ints(args.eval_simulations))
    by_stone = {
        stone: [r for r in results if r.opponent == "minimax" and r.simulations == low_sim and r.stones == stone]
        for stone in stones
    }
    scores = {
        stone: items[0].score_rate
        for stone, items in by_stone.items()
        if items
    }
    if not scores:
        return None
    weakest_stone = min(scores, key=scores.get)
    mean_score = sum(scores.values()) / len(scores)
    if scores[weakest_stone] < args.focus_threshold and mean_score - scores[weakest_stone] >= args.focus_margin:
        return weakest_stone
    return None


def append_csv(path: Path, rows: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    command_log = output_dir / "commands.log"
    stones = parse_csv_ints(args.stones)
    specs = build_eval_specs(args, stones)
    previous_checkpoint = Path(args.start_checkpoint) if args.start_checkpoint else None
    completed_games = read_completed_games(previous_checkpoint)
    best_score = float("-inf")
    best_checkpoint: str | None = None
    next_focus: int | None = None
    current_stone_weights = (
        format_stone_weights(parse_stone_weight_text(args.stone_weights, stones))
        if args.balance_stone_weights
        else args.stone_weights
    )

    print(f"writing overnight logs to {output_dir}")
    print(f"starting from completed_games={completed_games}")

    for block in range(1, args.blocks + 1):
        checkpoint = checkpoint_dir / f"{args.checkpoint_prefix}_block_{block:03d}.pt"
        target_games = completed_games + args.block_games
        train_simulations = scheduled_value(args.train_simulations_schedule, args.train_simulations, block)
        epochs = scheduled_value(args.epochs_schedule, args.epochs, block)
        train_command = build_train_command(
            args,
            checkpoint,
            previous_checkpoint,
            target_games,
            None if args.balance_stone_weights else next_focus,
            stones,
            train_simulations=train_simulations,
            epochs=epochs,
            stone_weights=current_stone_weights if args.balance_stone_weights else None,
        )
        if args.balance_stone_weights:
            focus_text = f"stone_weights={current_stone_weights}"
        elif next_focus is not None:
            focus_text = f"stones={next_focus}"
        elif args.stone_weights is not None:
            focus_text = f"stone_weights={args.stone_weights}"
        else:
            focus_text = f"stones={min(stones)}..{max(stones)}"
        print(
            f"\nblock {block}/{args.blocks}: train to {target_games} games "
            f"({focus_text}, simulations={train_simulations}, epochs={epochs})"
        )
        with command_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{now_iso()} TRAIN {command_text(train_command)}\n")
        if args.dry_run:
            print(command_text(train_command))
        else:
            subprocess.run(train_command, check=True)
        completed_games = target_games
        previous_checkpoint = checkpoint

        block_results: list[EvalResult] = []
        print(f"block {block}: evaluating {checkpoint}")
        for spec_index, spec in enumerate(specs):
            eval_seed = args.seed + block * 10_000 + spec_index
            eval_command = build_eval_command(args, checkpoint, spec, eval_seed)
            with command_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{now_iso()} EVAL {command_text(eval_command)}\n")
            if args.dry_run:
                print(command_text(eval_command))
                continue
            block_results.append(run_evaluation(args, block, checkpoint, spec, eval_seed))

        if args.dry_run:
            continue
        append_csv(metrics_path, block_results)
        block_score = checkpoint_score(args, block_results)
        if block_score > best_score:
            best_score = block_score
            best_checkpoint = str(output_dir / "best.pt")
            shutil.copy2(checkpoint, best_checkpoint)
            print(f"block {block}: new best score {best_score:.3f} -> {best_checkpoint}")
        if args.balance_stone_weights:
            current_stone_weights = balanced_stone_weights(args, stones, block_results, current_stone_weights)
            next_focus = None
            guidance = f"next block: stone_weights={current_stone_weights}"
        else:
            next_focus = choose_next_focus(args, stones, block_results)
            if next_focus is None:
                guidance = f"next block: mixed {min(stones)}..{max(stones)}"
            else:
                guidance = f"next block: focus stones={next_focus}"
        print(f"block {block}: coach score={block_score:.3f}; {guidance}")
        write_json(
            summary_path,
            {
                "updated_at": now_iso(),
                "best_score": best_score,
                "best_checkpoint": best_checkpoint,
                "completed_games": completed_games,
                "last_checkpoint": str(checkpoint),
                "next_focus": next_focus,
                "next_stone_weights": current_stone_weights if args.balance_stone_weights else None,
                "last_block_score": block_score,
                "last_train_simulations": train_simulations,
                "last_epochs": epochs,
                "last_block_results": [asdict(result) for result in block_results],
                "args": vars(args),
            },
        )


if __name__ == "__main__":
    main()
