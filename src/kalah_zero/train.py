from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from kalah_zero.encoding import encode_state
from kalah_zero.evaluate import choose_opening_plies, choose_stones, random_opening
from kalah_zero.game import GameState
from kalah_zero.mcts import MCTS, Evaluator


@dataclass(frozen=True, slots=True)
class TrainConfig:
    pits: int = 6
    stones: int = 4
    stones_min: int | None = None
    stones_max: int | None = None
    simulations: int = 50
    games_per_iteration: int = 8
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_capacity: int = 10_000
    temperature_moves: int = 12
    seed: int = 0
    use_batched_mcts: bool = False
    eval_batch_size: int = 32
    opening_plies: int = 0
    opening_plies_min: int | None = None
    opening_plies_max: int | None = None
    use_fast_game: bool = False
    model_type: str = "mlp"
    hidden_size: int = 128
    residual_blocks: int = 3


@dataclass(frozen=True, slots=True)
class TrainingSample:
    state: GameState
    policy: tuple[float, ...]
    value: float


@dataclass(slots=True)
class ReplayBuffer:
    capacity: int
    rng: random.Random = field(default_factory=random.Random)
    samples: list[TrainingSample] = field(default_factory=list)

    def add_many(self, samples: list[TrainingSample]) -> None:
        self.samples.extend(samples)
        overflow = len(self.samples) - self.capacity
        if overflow > 0:
            del self.samples[:overflow]

    def sample(self, batch_size: int) -> list[TrainingSample]:
        if not self.samples:
            raise ValueError("cannot sample from an empty replay buffer")
        if len(self.samples) <= batch_size:
            return list(self.samples)
        return self.rng.sample(self.samples, batch_size)

    def __len__(self) -> int:
        return len(self.samples)


def self_play_game(
    evaluator: Evaluator,
    config: TrainConfig,
    rng: random.Random | None = None,
    mcts_factory: Callable[[], MCTS] | None = None,
) -> list[TrainingSample]:
    rng = rng or random.Random(config.seed)
    if mcts_factory is None:
        mcts = MCTS(
            simulations=config.simulations,
            dirichlet_alpha=0.3,
            rng=rng,
        )
    else:
        mcts = mcts_factory()
    state_cls = GameState
    if config.use_fast_game:
        from kalah_zero.fast_game import FastGameState

        state_cls = FastGameState
    state = random_opening(
        choose_opening_plies(
            rng,
            opening_plies=config.opening_plies,
            opening_plies_min=config.opening_plies_min,
            opening_plies_max=config.opening_plies_max,
        ),
        rng,
        pits=config.pits,
        stones=choose_stones(
            rng,
            stones=config.stones,
            stones_min=config.stones_min,
            stones_max=config.stones_max,
        ),
        state_cls=state_cls,
    )
    trajectory: list[tuple[GameState, tuple[float, ...], int]] = []
    move_index = 0
    while not state.is_terminal():
        result = mcts.search(state, evaluator)
        temperature = 1.0 if move_index < config.temperature_moves else 0.0
        action = result.select_action(temperature=temperature, rng=rng)
        trajectory.append((state, tuple(result.policy), state.current_player))
        state = state.apply(action)
        move_index += 1

    return [
        TrainingSample(position, policy, state.reward_for_player(player))
        for position, policy, player in trajectory
    ]


def train_step(model, optimizer, batch: list[TrainingSample], l2_weight: float = 0.0) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if not batch:
        raise ValueError("batch must not be empty")

    model.train()
    states = torch.stack([encode_state(sample.state) for sample in batch])
    policy_targets = torch.tensor([sample.policy for sample in batch], dtype=torch.float32)
    value_targets = torch.tensor([sample.value for sample in batch], dtype=torch.float32)

    logits, values = model(states)
    policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    l2_loss = torch.zeros((), dtype=torch.float32)
    if l2_weight:
        for parameter in model.parameters():
            l2_loss = l2_loss + parameter.pow(2).sum()
        l2_loss = l2_weight * l2_loss
    loss = policy_loss + value_loss + l2_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "l2_loss": float(l2_loss.detach()),
    }
