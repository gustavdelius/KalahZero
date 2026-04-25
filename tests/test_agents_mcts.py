from __future__ import annotations

import random
import unittest

from kalah_zero.agents import GreedyAgent, MinimaxAgent, NoisyAgent, RandomAgent
from kalah_zero.batched_mcts import BatchedMCTS
from kalah_zero.evaluate import arena, choose_opening_plies, random_opening
from kalah_zero.game import GameState
from kalah_zero.mcts import MCTS, UniformEvaluator


class AgentAndMCTSTests(unittest.TestCase):
    def test_random_agent_selects_legal_action(self) -> None:
        state = GameState.new_game()
        agent = RandomAgent(random.Random(0))

        self.assertIn(agent.select_action(state), state.legal_actions())

    def test_noisy_agent_can_delegate_or_choose_randomly(self) -> None:
        state = GameState.new_game()
        greedy = GreedyAgent()

        self.assertEqual(NoisyAgent(greedy, epsilon=0.0, rng=random.Random(0)).select_action(state), 2)
        self.assertIn(NoisyAgent(greedy, epsilon=1.0, rng=random.Random(0)).select_action(state), state.legal_actions())

    def test_greedy_agent_prefers_extra_store_move_from_start(self) -> None:
        state = GameState.new_game()
        agent = GreedyAgent()

        self.assertEqual(agent.select_action(state), 2)

    def test_minimax_agent_selects_legal_action(self) -> None:
        state = GameState.new_game()
        agent = MinimaxAgent(depth=2)

        self.assertIn(agent.select_action(state), state.legal_actions())

    def test_greedy_agent_beats_random_with_fixed_seed(self) -> None:
        result = arena(GreedyAgent(), RandomAgent(random.Random(0)), games=6, seed=0)

        self.assertGreaterEqual(result.wins_0, 5)

    def test_arena_reports_progress_after_each_game(self) -> None:
        progress: list[tuple[int, int]] = []

        arena(
            GreedyAgent(),
            RandomAgent(random.Random(0)),
            games=3,
            seed=0,
            on_game_complete=lambda completed, result: progress.append((completed, result.games)),
        )

        self.assertEqual(progress, [(1, 1), (2, 2), (3, 3)])

    def test_random_opening_applies_requested_number_of_moves(self) -> None:
        state = random_opening(plies=4, rng=random.Random(0))

        self.assertNotEqual(state, GameState.new_game())
        self.assertEqual(sum(state.board), GameState.new_game().total_stones)

    def test_choose_opening_plies_supports_ranges(self) -> None:
        rng = random.Random(0)
        samples = [choose_opening_plies(rng, opening_plies_min=0, opening_plies_max=8) for _ in range(20)]

        self.assertTrue(all(0 <= sample <= 8 for sample in samples))
        self.assertGreater(len(set(samples)), 1)

    def test_arena_accepts_random_openings(self) -> None:
        result = arena(
            GreedyAgent(),
            RandomAgent(random.Random(0)),
            games=4,
            seed=0,
            opening_plies=3,
        )

        self.assertEqual(result.games, 4)
        self.assertEqual(result.wins_0 + result.wins_1 + result.draws, 4)

    def test_arena_accepts_random_opening_ranges(self) -> None:
        result = arena(
            GreedyAgent(),
            RandomAgent(random.Random(0)),
            games=4,
            seed=0,
            opening_plies_min=0,
            opening_plies_max=4,
        )

        self.assertEqual(result.games, 4)
        self.assertEqual(result.wins_0 + result.wins_1 + result.draws, 4)

    def test_mcts_returns_visit_policy_over_legal_actions(self) -> None:
        state = GameState.new_game()
        mcts = MCTS(simulations=20)

        result = mcts.search(state, UniformEvaluator())

        self.assertEqual(sum(result.visits), 20)
        self.assertAlmostEqual(sum(result.policy), 1.0)
        self.assertTrue(all(result.policy[action] > 0 for action in state.legal_actions()))

    def test_mcts_masks_illegal_policy_entries(self) -> None:
        state = GameState(
            board=(0, 0, 1, 0, 0, 1, 0, 0, 0, 5, 0, 0, 1, 0),
            current_player=0,
            pits=6,
        )
        mcts = MCTS(simulations=10)

        result = mcts.search(state, UniformEvaluator())

        self.assertEqual(result.policy[0], 0.0)
        self.assertEqual(result.policy[1], 0.0)
        self.assertGreater(result.policy[2] + result.policy[5], 0.0)

    def test_batched_mcts_returns_visit_policy_over_legal_actions(self) -> None:
        state = GameState.new_game()
        mcts = BatchedMCTS(simulations=20, batch_size=4)

        result = mcts.search(state, UniformEvaluator())

        self.assertEqual(sum(result.visits), 20)
        self.assertAlmostEqual(sum(result.policy), 1.0)
        self.assertTrue(all(result.policy[action] > 0 for action in state.legal_actions()))

    def test_batched_mcts_uses_batch_evaluator_when_available(self) -> None:
        class CountingEvaluator(UniformEvaluator):
            def __init__(self) -> None:
                self.batch_calls = 0

            def evaluate_batch(self, states: list[GameState]) -> list[tuple[list[float], float]]:
                self.batch_calls += 1
                return [self.evaluate(state) for state in states]

        evaluator = CountingEvaluator()
        mcts = BatchedMCTS(simulations=8, batch_size=4)

        mcts.search(GameState.new_game(), evaluator)

        self.assertGreater(evaluator.batch_calls, 0)


if __name__ == "__main__":
    unittest.main()
