#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys

import _path  # noqa: F401
from kalah_zero.agents import GreedyAgent, MCTSAgent, MinimaxAgent, NoisyAgent, RandomAgent
from kalah_zero.evaluate import arena
from kalah_zero.mcts import MCTS, UniformEvaluator


AGENT_CHOICES = ["random", "greedy", "minimax", "mcts", "noisy-greedy", "noisy-minimax", "noisy-mcts"]


def build_search(simulations: int, use_batched_mcts: bool, eval_batch_size: int):
    if use_batched_mcts:
        from kalah_zero.batched_mcts import BatchedMCTS

        return BatchedMCTS(simulations=simulations, batch_size=eval_batch_size)
    return MCTS(simulations=simulations)


def build_agent(
    name: str,
    seed: int,
    checkpoint: str | None = None,
    simulations: int = 100,
    use_batched_mcts: bool = False,
    eval_batch_size: int = 32,
    noise_prob: float = 0.1,
):
    noisy = name.startswith("noisy-")
    base_name = name.removeprefix("noisy-")
    if checkpoint is not None:
        from kalah_zero.network import NeuralEvaluator, load_checkpoint

        model, _ = load_checkpoint(checkpoint)
        agent = MCTSAgent(
            build_search(simulations, use_batched_mcts, eval_batch_size),
            NeuralEvaluator(model),
            temperature=0.0,
        )
        return NoisyAgent(agent, epsilon=noise_prob, rng=random.Random(seed)) if noisy else agent
    if base_name == "random":
        agent = RandomAgent(random.Random(seed))
    elif base_name == "greedy":
        agent = GreedyAgent()
    elif base_name == "minimax":
        agent = MinimaxAgent(depth=6)
    elif base_name == "mcts":
        agent = MCTSAgent(
            build_search(simulations, use_batched_mcts, eval_batch_size),
            UniformEvaluator(),
            temperature=0.0,
        )
    else:
        raise ValueError(f"unknown agent {name}")
    return NoisyAgent(agent, epsilon=noise_prob, rng=random.Random(seed)) if noisy else agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two Kalah agents.")
    parser.add_argument("--agent-a", choices=AGENT_CHOICES, default="greedy")
    parser.add_argument("--agent-b", choices=AGENT_CHOICES, default="random")
    parser.add_argument("--checkpoint-a", help="Use a neural checkpoint for agent A.")
    parser.add_argument("--checkpoint-b", help="Use a neural checkpoint for agent B.")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--batched-mcts", action="store_true", help="Batch MCTS leaf evaluations.")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=0,
        help="Play this many random legal moves before evaluation. Openings are reused with agents swapped.",
    )
    parser.add_argument(
        "--opening-plies-min",
        type=int,
        help="Minimum random opening length. Openings are reused with agents swapped.",
    )
    parser.add_argument(
        "--opening-plies-max",
        type=int,
        help="Maximum random opening length. Openings are reused with agents swapped.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.1,
        help="Random-move probability for noisy-* agents.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def show_progress(completed: int, partial_result) -> None:
        print(
            f"\rcompleted {completed}/{args.games} games "
            f"({partial_result.wins_0}-{partial_result.wins_1}-{partial_result.draws})",
            end="",
            file=sys.stderr,
            flush=True,
        )

    result = arena(
        build_agent(
            args.agent_a,
            args.seed,
            args.checkpoint_a,
            args.simulations,
            args.batched_mcts,
            args.eval_batch_size,
            args.noise_prob,
        ),
        build_agent(
            args.agent_b,
            args.seed + 1,
            args.checkpoint_b,
            args.simulations,
            args.batched_mcts,
            args.eval_batch_size,
            args.noise_prob,
        ),
        games=args.games,
        seed=args.seed,
        opening_plies=args.opening_plies,
        opening_plies_min=args.opening_plies_min,
        opening_plies_max=args.opening_plies_max,
        on_game_complete=show_progress,
    )
    print(file=sys.stderr)
    label_a = args.checkpoint_a or args.agent_a
    label_b = args.checkpoint_b or args.agent_b
    search_info = f"simulations={args.simulations}"
    if args.batched_mcts:
        search_info += f", batched_mcts=True, eval_batch_size={args.eval_batch_size}"
    using_opening_range = args.opening_plies_min is not None or args.opening_plies_max is not None
    if args.opening_plies > 0 and not using_opening_range:
        search_info += f", opening_plies={args.opening_plies}"
    if using_opening_range:
        opening_min = 0 if args.opening_plies_min is None else args.opening_plies_min
        opening_max = opening_min if args.opening_plies_max is None else args.opening_plies_max
        search_info += f", opening_plies={opening_min}..{opening_max}"
    if args.agent_a.startswith("noisy-") or args.agent_b.startswith("noisy-"):
        search_info += f", noise_prob={args.noise_prob}"
    print(
        f"{label_a} vs {label_b}: "
        f"{result.wins_0}-{result.wins_1}-{result.draws} "
        f"(win rate {result.win_rate_0:.3f}, {search_info})"
    )


if __name__ == "__main__":
    main()
