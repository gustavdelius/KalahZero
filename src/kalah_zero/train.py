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
    """All hyperparameters for one training run, bundled into a single frozen object.

    Frozen means the fields cannot be changed after construction, which makes it
    safe to pass the config to multiple functions without worrying about accidental
    mutation.
    """

    pits: int = 6                        # number of pits per side
    stones: int = 4                      # starting stones per pit (used when stones_min/max are None)
    stones_min: int | None = None        # if set, sample stones uniformly from [stones_min, stones_max]
    stones_max: int | None = None
    stone_weights: tuple[tuple[int, float], ...] | None = None  # weighted starting-stone distribution
    simulations: int = 50                # MCTS simulations per move
    games_per_iteration: int = 8         # self-play games to generate before each training update
    batch_size: int = 64                 # number of samples drawn from the replay buffer per gradient step
    epochs: int = 2                      # gradient steps per training iteration
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4           # L2 penalty applied by the optimizer (Adam weight decay)
    replay_capacity: int = 10_000        # maximum number of samples kept in the replay buffer
    temperature_moves: int = 12          # moves at the start of each game played with temperature=1
    seed: int = 0
    use_batched_mcts: bool = False       # use the batched GPU MCTS implementation instead of the plain one
    eval_batch_size: int = 32            # batch size used by the batched evaluator
    opening_plies: int = 0              # fixed number of random opening moves before search begins
    opening_plies_min: int | None = None # if set, sample opening length uniformly from [min, max]
    opening_plies_max: int | None = None
    use_fast_game: bool = False          # use the C-extension FastGameState for speed
    model_type: str = "mlp"             # network architecture: "mlp" or "residual"
    hidden_size: int = 128              # width of each hidden layer
    residual_blocks: int = 3            # number of residual blocks (only used when model_type="residual")

    def __post_init__(self) -> None:
        if self.stone_weights is None:
            return
        normalized: list[tuple[int, float]] = []
        for stones, weight in self.stone_weights:
            stone_count = int(stones)
            stone_weight = float(weight)
            if stone_count < 0:
                raise ValueError("stone weights must use non-negative stone counts")
            if stone_weight <= 0:
                raise ValueError("stone weights must be positive")
            normalized.append((stone_count, stone_weight))
        if not normalized:
            raise ValueError("stone_weights must not be empty")
        object.__setattr__(self, "stone_weights", tuple(normalized))


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One training example produced by self-play.

    Collected at a single move during a game; the value is filled in
    retrospectively once the game has finished.
    """

    state: GameState           # board position before the move
    policy: tuple[float, ...]  # MCTS visit-count distribution (the search policy π)
    value: float               # final game outcome for the player who was to move (+1, 0, or -1)


@dataclass(slots=True)
class ReplayBuffer:
    """Fixed-capacity circular store of recent TrainingSamples.

    When the buffer is full, the oldest samples are dropped to make room for
    new ones, so the network always trains on recent experience.
    """

    capacity: int
    rng: random.Random = field(default_factory=random.Random)
    samples: list[TrainingSample] = field(default_factory=list)

    def add_many(self, samples: list[TrainingSample]) -> None:
        """Append samples and evict the oldest ones if capacity is exceeded."""
        self.samples.extend(samples)
        overflow = len(self.samples) - self.capacity
        if overflow > 0:
            del self.samples[:overflow]

    def sample(self, batch_size: int) -> list[TrainingSample]:
        """Return a random batch of up to batch_size samples without replacement."""
        if not self.samples:
            raise ValueError("cannot sample from an empty replay buffer")
        if len(self.samples) <= batch_size:
            return list(self.samples)
        return self.rng.sample(self.samples, batch_size)

    def __len__(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self.samples)


def self_play_game(
    evaluator: Evaluator,
    config: TrainConfig,
    rng: random.Random | None = None,
    mcts_factory: Callable[[], MCTS] | None = None,
) -> list[TrainingSample]:
    """Play one full self-play game and return the training samples it produced.

    The game starts from a (possibly randomly chosen) opening position. At each
    move the current network guides MCTS, and the resulting visit distribution
    is recorded as the policy target. Once the game ends, the final outcome is
    attached to every recorded position as the value target.

    mcts_factory is accepted as a parameter so callers can supply a pre-configured
    MCTS instance (e.g. one that shares a neural network across games). When it is
    None, a default MCTS is constructed from the config.
    """
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
    # Apply random opening moves to diversify the starting position.
    # These moves are not recorded as training samples; they only set the scene.
    state = random_opening(
        choose_opening_plies(
            rng,
            opening_plies=config.opening_plies,
            opening_plies_min=config.opening_plies_min,
            opening_plies_max=config.opening_plies_max,
        ),
        rng,
        pits=config.pits,
        stones=choose_training_stones(
            rng,
            config,
        ),
        state_cls=state_cls,
    )
    # trajectory accumulates (state, search_policy, player_to_move) for each move.
    # The value target is not yet known here; it is filled in after the game ends.
    trajectory: list[tuple[GameState, tuple[float, ...], int]] = []
    move_index = 0
    while not state.is_terminal():
        result = mcts.search(state, evaluator)
        # Use temperature=1 for the first few moves to encourage variety in the opening,
        # then switch to greedy (temperature=0) so the outcome is more decisive.
        temperature = 1.0 if move_index < config.temperature_moves else 0.0
        action = result.select_action(temperature=temperature, rng=rng)
        trajectory.append((state, tuple(result.policy), state.current_player))
        state = state.apply(action)
        move_index += 1

    # Retrospectively assign the final outcome to every position in the game.
    # state.reward_for_player returns the result from the perspective of the player
    # who was to move at that position, which is what the value head must learn to predict.
    return [
        TrainingSample(position, policy, state.reward_for_player(player))
        for position, policy, player in trajectory
    ]


def choose_training_stones(rng: random.Random, config: TrainConfig) -> int:
    if config.stone_weights is None:
        return choose_stones(
            rng,
            stones=config.stones,
            stones_min=config.stones_min,
            stones_max=config.stones_max,
        )
    choices = [stones for stones, _ in config.stone_weights]
    weights = [weight for _, weight in config.stone_weights]
    return rng.choices(choices, weights=weights, k=1)[0]


def train_step(model, optimizer, batch: list[TrainingSample], l2_weight: float = 0.0) -> dict[str, float]:
    """Run one gradient update on the model and return the component losses.

    The total loss has three terms:
    - policy loss: cross-entropy between the network's move probabilities and the MCTS targets.
    - value loss: mean squared error between the network's outcome prediction and the game result.
    - l2_loss: optional explicit L2 weight penalty (separate from the optimizer's weight_decay).
    """
    import torch
    import torch.nn.functional as F

    if not batch:
        raise ValueError("batch must not be empty")

    model.train()  # enable dropout and batch-norm training behaviour if present
    states = torch.stack([encode_state(sample.state) for sample in batch])
    policy_targets = torch.tensor([sample.policy for sample in batch], dtype=torch.float32)
    value_targets = torch.tensor([sample.value for sample in batch], dtype=torch.float32)

    logits, values = model(states)
    # Cross-entropy loss: -sum(π * log p). log_softmax is numerically more stable than
    # taking softmax first and then log.
    policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    l2_loss = torch.zeros((), dtype=torch.float32)
    if l2_weight:
        # Accumulate the squared norm of every parameter tensor.
        for parameter in model.parameters():
            l2_loss = l2_loss + parameter.pow(2).sum()
        l2_loss = l2_weight * l2_loss
    loss = policy_loss + value_loss + l2_loss

    optimizer.zero_grad()
    loss.backward()   # compute gradients for all parameters
    optimizer.step()  # apply the gradient update

    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "l2_loss": float(l2_loss.detach()),
    }
