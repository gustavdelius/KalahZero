from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kalah_zero.encoding import encode_features, input_size
from kalah_zero.game import GameState
from kalah_zero.mcts import UniformEvaluator
from kalah_zero.train import TrainConfig, ReplayBuffer, self_play_game


class EncodingAndTrainingTests(unittest.TestCase):
    def test_encode_features_has_expected_size(self) -> None:
        state = GameState.new_game()

        features = encode_features(state)

        self.assertEqual(len(features), input_size(state.pits))
        self.assertEqual(features[-1], 1.0)

    def test_self_play_generates_training_samples(self) -> None:
        config = TrainConfig(simulations=5, stones=1, temperature_moves=2)

        samples = self_play_game(UniformEvaluator(), config)

        self.assertGreater(len(samples), 0)
        self.assertTrue(all(len(sample.policy) == config.pits for sample in samples))
        self.assertTrue(all(sample.value in (-1.0, 0.0, 1.0) for sample in samples))

    def test_self_play_accepts_batched_mcts_factory(self) -> None:
        from kalah_zero.batched_mcts import BatchedMCTS

        config = TrainConfig(simulations=4, stones=1, temperature_moves=2)

        samples = self_play_game(
            UniformEvaluator(),
            config,
            mcts_factory=lambda: BatchedMCTS(simulations=config.simulations, batch_size=2),
        )

        self.assertGreater(len(samples), 0)

    def test_self_play_can_start_from_random_opening(self) -> None:
        config = TrainConfig(simulations=3, stones=2, opening_plies=3)

        samples = self_play_game(UniformEvaluator(), config)

        self.assertGreater(len(samples), 0)
        self.assertNotEqual(samples[0].state, GameState.new_game(stones=config.stones))

    def test_self_play_can_sample_random_opening_depths(self) -> None:
        config = TrainConfig(simulations=3, stones=2, opening_plies_min=0, opening_plies_max=4)

        samples = self_play_game(UniformEvaluator(), config)

        self.assertGreater(len(samples), 0)

    def test_replay_buffer_respects_capacity(self) -> None:
        config = TrainConfig(simulations=2, stones=1)
        samples = self_play_game(UniformEvaluator(), config)
        buffer = ReplayBuffer(capacity=3)

        buffer.add_many(samples)

        self.assertLessEqual(len(buffer), 3)
        self.assertGreater(len(buffer.sample(2)), 0)


class OptionalTorchTests(unittest.TestCase):
    def test_network_shapes_when_torch_is_installed(self) -> None:
        try:
            import torch  # noqa: F401
            from kalah_zero.encoding import encode_state
            from kalah_zero.network import KalahNet, NeuralEvaluator
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        state = GameState.new_game()
        model = KalahNet(pits=state.pits)
        logits, value = model(encode_state(state).unsqueeze(0))
        evaluator = NeuralEvaluator(model)
        policy, scalar = evaluator.evaluate(state)
        batch_results = evaluator.evaluate_batch([state, state.apply(2)])

        self.assertEqual(tuple(logits.shape), (1, state.pits))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(len(policy), state.pits)
        self.assertTrue(-1.0 <= scalar <= 1.0)
        self.assertEqual(len(batch_results), 2)
        self.assertTrue(all(len(item[0]) == state.pits for item in batch_results))

    def test_training_checkpoint_round_trip(self) -> None:
        try:
            import random
            import sys

            import torch
            from kalah_zero.network import KalahNet

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
            from scripts.train import build_parser, config_from_args, load_training_checkpoint, save_training_checkpoint
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        rng = random.Random(123)
        config = TrainConfig(simulations=1, games_per_iteration=3, batch_size=2, epochs=1, seed=123)
        model = KalahNet(pits=config.pits)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        samples = self_play_game(UniformEvaluator(), config, rng)
        buffer = ReplayBuffer(config.replay_capacity, rng=rng)
        buffer.add_many(samples)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model,
                optimizer,
                buffer,
                rng,
                config,
                completed_games=2,
                reason="test",
            )
            loaded_model, loaded_optimizer, loaded_buffer, loaded_rng, loaded_config, completed_games = (
                load_training_checkpoint(path)
            )

        self.assertEqual(loaded_model.pits, config.pits)
        self.assertIsNotNone(loaded_optimizer)
        self.assertIsNotNone(loaded_rng)
        self.assertEqual(loaded_config.simulations, config.simulations)
        self.assertEqual(loaded_config.opening_plies, config.opening_plies)
        self.assertEqual(completed_games, 2)
        self.assertEqual(len(loaded_buffer), len(buffer))

        parser = build_parser()
        args = parser.parse_args([
            "--replay-capacity",
            "50000",
            "--opening-plies-min",
            "0",
            "--opening-plies-max",
            "8",
        ])
        updated_config = config_from_args(args, loaded_config)

        self.assertEqual(updated_config.replay_capacity, 50_000)
        self.assertEqual(updated_config.opening_plies_min, 0)
        self.assertEqual(updated_config.opening_plies_max, 8)


if __name__ == "__main__":
    unittest.main()
