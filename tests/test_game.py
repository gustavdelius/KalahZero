from __future__ import annotations

import unittest

from kalah_zero.game import GameState


class GameStateTests(unittest.TestCase):
    def test_new_game_has_expected_legal_actions(self) -> None:
        state = GameState.new_game()

        self.assertEqual(state.legal_actions(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(state.store_for(0), 0)
        self.assertEqual(state.store_for(1), 0)
        self.assertEqual(state.total_stones, 48)

    def test_sowing_and_extra_turn(self) -> None:
        state = GameState.new_game()

        child = state.apply(2)

        self.assertEqual(child.pits_for(0), (4, 4, 0, 5, 5, 5))
        self.assertEqual(child.store_for(0), 1)
        self.assertEqual(child.current_player, 0)

    def test_sowing_skips_opponent_store(self) -> None:
        state = GameState(
            board=(1, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 1, 0),
            current_player=0,
            pits=6,
        )

        child = state.apply(5)

        self.assertEqual(child.store_for(1), 0)
        self.assertEqual(child.total_stones, state.total_stones)

    def test_capture_from_opposite_pit_gives_another_turn(self) -> None:
        state = GameState(
            board=(0, 0, 1, 0, 0, 1, 0, 0, 0, 5, 0, 0, 1, 0),
            current_player=0,
            pits=6,
        )

        child = state.apply(2)

        self.assertEqual(child.board[3], 0)
        self.assertEqual(child.board[9], 0)
        self.assertEqual(child.store_for(0), 6)
        self.assertEqual(child.current_player, 0)

    def test_endgame_sweeps_remaining_stones(self) -> None:
        state = GameState(
            board=(0, 0, 0, 0, 0, 1, 10, 1, 2, 3, 4, 5, 6, 7),
            current_player=0,
            pits=6,
        )

        child = state.apply(5)

        self.assertTrue(child.is_terminal())
        self.assertEqual(child.pits_for(0), (0, 0, 0, 0, 0, 0))
        self.assertEqual(child.pits_for(1), (0, 0, 0, 0, 0, 0))
        self.assertEqual(child.store_for(0), 11)
        self.assertEqual(child.store_for(1), 28)

    def test_illegal_move_raises(self) -> None:
        state = GameState.new_game().apply(2)

        with self.assertRaises(ValueError):
            state.apply(2)

    def test_reward_is_terminal_score_sign(self) -> None:
        state = GameState(
            board=(0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 18),
            current_player=0,
            pits=6,
        )

        self.assertEqual(state.reward_for_player(0), 1.0)
        self.assertEqual(state.reward_for_player(1), -1.0)


class FastGameStateTests(unittest.TestCase):
    def test_fast_game_matches_python_for_known_rules(self) -> None:
        try:
            from kalah_zero.fast_game import FastGameState
        except ImportError:
            self.skipTest("C++ fast game extension is not built")

        positions = [
            GameState.new_game(),
            GameState(board=(0, 0, 1, 0, 0, 1, 0, 0, 0, 5, 0, 0, 1, 0), current_player=0),
            GameState(board=(0, 0, 0, 0, 0, 1, 10, 1, 2, 3, 4, 5, 6, 7), current_player=0),
        ]
        for state in positions:
            fast = FastGameState(state.board, state.current_player, state.pits)
            self.assertEqual(fast.legal_actions(), state.legal_actions())
            for action in state.legal_actions():
                expected = state.apply(action)
                actual = fast.apply(action)
                self.assertEqual(actual.board, expected.board)
                self.assertEqual(actual.current_player, expected.current_player)

    def test_fast_game_matches_python_for_random_play(self) -> None:
        try:
            from kalah_zero.fast_game import FastGameState
        except ImportError:
            self.skipTest("C++ fast game extension is not built")

        import random

        rng = random.Random(0)
        python_state = GameState.new_game()
        fast_state = FastGameState.new_game()
        for _ in range(50):
            self.assertEqual(fast_state.board, python_state.board)
            self.assertEqual(fast_state.current_player, python_state.current_player)
            legal = python_state.legal_actions()
            self.assertEqual(fast_state.legal_actions(), legal)
            if not legal:
                break
            action = rng.choice(legal)
            python_state = python_state.apply(action)
            fast_state = fast_state.apply(action)


if __name__ == "__main__":
    unittest.main()
