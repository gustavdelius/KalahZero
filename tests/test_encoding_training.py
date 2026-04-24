from __future__ import annotations

import unittest

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

        self.assertEqual(tuple(logits.shape), (1, state.pits))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(len(policy), state.pits)
        self.assertTrue(-1.0 <= scalar <= 1.0)


if __name__ == "__main__":
    unittest.main()

