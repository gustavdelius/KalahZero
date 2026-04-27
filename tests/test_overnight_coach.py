from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from overnight_coach import (  # noqa: E402
    EvalResult,
    balanced_stone_weights,
    build_parser,
    build_train_command,
    checkpoint_score,
    choose_next_focus,
    format_stone_weights,
    parse_result_line,
    parse_stone_weight_text,
    scheduled_value,
    score_rate,
)


class OvernightCoachTests(unittest.TestCase):
    def test_parse_result_line_ignores_progress_output(self) -> None:
        text = """
completed 200/200 games (90-105-5)
checkpoints/model.pt vs minimax: 90-105-5 (win rate 0.450, simulations=100)
"""

        self.assertEqual(parse_result_line(text), (90, 105, 5, 0.45))

    def test_score_rate_counts_draws_as_half(self) -> None:
        self.assertEqual(score_rate(wins=9, draws=2, games=20), 0.5)

    def test_adaptive_curriculum_focuses_weakest_low_sim_stone(self) -> None:
        args = build_parser().parse_args([])
        results = [
            self._result(stones=4, simulations=25, score=0.60),
            self._result(stones=5, simulations=25, score=0.52),
            self._result(stones=6, simulations=25, score=0.30),
        ]

        self.assertEqual(choose_next_focus(args, [4, 5, 6], results), 6)

    def test_adaptive_curriculum_uses_mixed_when_scores_are_close(self) -> None:
        args = build_parser().parse_args([])
        results = [
            self._result(stones=4, simulations=25, score=0.46),
            self._result(stones=5, simulations=25, score=0.45),
            self._result(stones=6, simulations=25, score=0.44),
        ]

        self.assertIsNone(choose_next_focus(args, [4, 5, 6], results))

    def test_checkpoint_score_penalizes_low_simulation_failures(self) -> None:
        args = build_parser().parse_args([])
        weak_low_sim = [
            self._result(stones=4, simulations=25, score=0.30),
            self._result(stones=5, simulations=25, score=0.30),
            self._result(stones=6, simulations=25, score=0.30),
            self._result(stones=4, simulations=150, score=0.90),
            self._result(stones=5, simulations=150, score=0.90),
            self._result(stones=6, simulations=150, score=0.90),
        ]
        steady = [
            self._result(stones=4, simulations=25, score=0.50),
            self._result(stones=5, simulations=25, score=0.50),
            self._result(stones=6, simulations=25, score=0.50),
            self._result(stones=4, simulations=150, score=0.60),
            self._result(stones=5, simulations=150, score=0.60),
            self._result(stones=6, simulations=150, score=0.60),
        ]

        self.assertGreater(checkpoint_score(args, steady), checkpoint_score(args, weak_low_sim))

    def test_train_command_resumes_to_total_target_games(self) -> None:
        args = build_parser().parse_args([
            "--block-games",
            "500",
            "--eval-batch-size",
            "8",
            "--no-fast-game",
        ])

        command = build_train_command(
            args,
            checkpoint=Path("runs/block2.pt"),
            previous_checkpoint=Path("runs/block1.pt"),
            target_games=1500,
            focus_stone=6,
            stones=[4, 5, 6],
        )

        self.assertIn("--resume", command)
        self.assertIn("runs/block1.pt", command)
        self.assertIn("--games", command)
        self.assertIn("1500", command)
        self.assertIn("--stones", command)
        self.assertIn("6", command)
        self.assertNotIn("--stones-min", command)

    def test_train_command_can_use_weighted_stone_distribution(self) -> None:
        args = build_parser().parse_args(["--stone-weights", "4:1,5:1,6:2"])

        command = build_train_command(
            args,
            checkpoint=Path("runs/block1.pt"),
            previous_checkpoint=None,
            target_games=1000,
            focus_stone=None,
            stones=[4, 5, 6],
        )

        self.assertIn("--stone-weights", command)
        self.assertIn("4:1,5:1,6:2", command)
        self.assertNotIn("--stones-min", command)

    def test_train_command_accepts_dynamic_stone_weights(self) -> None:
        args = build_parser().parse_args(["--balance-stone-weights"])

        command = build_train_command(
            args,
            checkpoint=Path("runs/block1.pt"),
            previous_checkpoint=None,
            target_games=1000,
            focus_stone=None,
            stones=[4, 5, 6],
            stone_weights="4:0.75,5:1,6:1.25",
        )

        self.assertIn("--stone-weights", command)
        self.assertIn("4:0.75,5:1,6:1.25", command)
        self.assertNotIn("--stones-min", command)

    def test_balanced_stone_weights_increase_underperforming_stones(self) -> None:
        args = build_parser().parse_args(["--eval-simulations", "25"])
        results = [
            self._result(stones=4, simulations=25, score=0.70),
            self._result(stones=5, simulations=25, score=0.50),
            self._result(stones=6, simulations=25, score=0.25),
        ]

        weights = parse_stone_weight_text(
            balanced_stone_weights(args, [4, 5, 6], results, "4:1,5:1,6:1"),
            [4, 5, 6],
        )

        self.assertLess(weights[4], weights[5])
        self.assertLess(weights[5], weights[6])

    def test_balanced_stone_weights_do_not_compound_extreme_previous_weights(self) -> None:
        args = build_parser().parse_args(["--eval-simulations", "25,50,100,150"])
        results = [
            self._result(stones=4, simulations=25, score=0.345),
            self._result(stones=5, simulations=25, score=0.160),
            self._result(stones=6, simulations=25, score=0.100),
            self._result(stones=4, simulations=50, score=0.325),
            self._result(stones=5, simulations=50, score=0.230),
            self._result(stones=6, simulations=50, score=0.225),
            self._result(stones=4, simulations=100, score=0.375),
            self._result(stones=5, simulations=100, score=0.240),
            self._result(stones=6, simulations=100, score=0.300),
            self._result(stones=4, simulations=150, score=0.365),
            self._result(stones=5, simulations=150, score=0.320),
            self._result(stones=6, simulations=150, score=0.275),
        ]

        weights = parse_stone_weight_text(
            balanced_stone_weights(args, [4, 5, 6], results, "4:0.25,5:0.25,6:2.63"),
            [4, 5, 6],
        )

        self.assertGreater(weights[5], weights[4])
        self.assertLess(weights[6], 2.0)
        self.assertGreater(weights[5], 0.25)

    def test_balanced_stone_weights_can_use_fresh_score_based_targets(self) -> None:
        args = build_parser().parse_args([
            "--eval-simulations",
            "25",
            "--stone-weight-smoothing",
            "0",
        ])
        results = [
            self._result(stones=4, simulations=25, score=0.60),
            self._result(stones=5, simulations=25, score=0.30),
            self._result(stones=6, simulations=25, score=0.30),
        ]

        weights = parse_stone_weight_text(
            balanced_stone_weights(args, [4, 5, 6], results, "4:0.25,5:0.25,6:3"),
            [4, 5, 6],
        )

        self.assertAlmostEqual(weights[4], 0.6, places=1)
        self.assertAlmostEqual(weights[5], 1.2, places=1)
        self.assertAlmostEqual(weights[6], 1.2, places=1)

    def test_stone_weight_text_defaults_missing_stones_to_one(self) -> None:
        weights = parse_stone_weight_text("6:2", [4, 5, 6])

        self.assertEqual(weights, {4: 1.0, 5: 1.0, 6: 2.0})
        self.assertEqual(format_stone_weights(weights), "4:1,5:1,6:2")

    def test_scheduled_value_repeats_last_item(self) -> None:
        self.assertEqual(scheduled_value("250,150,100", fallback=300, block=1), 250)
        self.assertEqual(scheduled_value("250,150,100", fallback=300, block=3), 100)
        self.assertEqual(scheduled_value("250,150,100", fallback=300, block=8), 100)
        self.assertEqual(scheduled_value(None, fallback=300, block=8), 300)

    def _result(self, stones: int, simulations: int, score: float) -> EvalResult:
        games = 100
        wins = int(score * games)
        return EvalResult(
            block=1,
            checkpoint="checkpoint.pt",
            opponent="minimax",
            stones=stones,
            simulations=simulations,
            games=games,
            wins=wins,
            losses=games - wins,
            draws=0,
            win_rate=wins / games,
            score_rate=score,
            noise_prob=None,
            timestamp="2026-04-25T00:00:00+00:00",
            command="",
        )


if __name__ == "__main__":
    unittest.main()
