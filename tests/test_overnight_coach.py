from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from overnight_coach import (  # noqa: E402
    EvalResult,
    build_parser,
    build_train_command,
    checkpoint_score,
    choose_next_focus,
    parse_result_line,
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
