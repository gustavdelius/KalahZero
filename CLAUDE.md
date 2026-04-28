# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

KalahZero is a teaching implementation of AlphaZero-style self-play learning for the board game Kalah. The codebase is intentionally compact so that learners can read the code alongside the tutorials in `tutorials/`.

## Commands

### Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

### Tests
```bash
python -m pytest                        # run all tests
python -m pytest tests/test_game.py     # run a single test file
python -m pytest -k test_legal_actions  # run a single test by name
```

### Training
```bash
python scripts/train.py --games 4 --simulations 25 --epochs 2       # quick smoke test
python scripts/train.py --games 300 --simulations 75 --epochs 1 --output checkpoints/my.pt
python scripts/train.py --resume checkpoints/my.pt                   # resume (--games = total, not additional)
python scripts/train.py --games 300 --simulations 150 --batched-mcts --eval-batch-size 32  # faster on CPU
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint-a checkpoints/my.pt --agent-b minimax --simulations 200
python scripts/evaluate.py --checkpoint-a checkpoints/my.pt --agent-b noisy-minimax --noise-prob 0.1
```

### Optional C++ game engine (speeds up high simulation counts)
```bash
python setup.py build_ext --inplace     # builds kalah_zero/_fast_game.so
```
Pass `--fast-game` to `train.py` or `evaluate.py` after building.

### Browser game
```bash
python scripts/export_browser_agent.py checkpoints/my.pt web/residual_depth.json
quarto preview    # opens the tutorial site with the Play page
quarto render     # builds _site/ for deployment
```

### Overnight coaching
```bash
python scripts/overnight_coach.py --output-dir runs/my-run --blocks 8 --block-games 500 \
  --train-simulations-schedule 250,250,200,200,150,150,100,100 --eval-games 100
```
Writes `metrics.csv`, `summary.json`, and `best.pt` to the output directory.

## Architecture

### Core library (`src/kalah_zero/`)

| Module | Purpose |
|---|---|
| `game.py` | Pure-Python immutable `GameState` (frozen dataclass). All game rules live here. |
| `fast_game.py` | Thin Python wrapper around the optional C++ extension `_fast_game.cpp`; drop-in for `GameState` in hot loops. |
| `encoding.py` | Converts a `GameState` to a flat float tensor for the network. The canonical layout is `[own pits, opponent pits reversed, stores, bias]`. The `ENCODING_VERSION` constant gates checkpoint compatibility. |
| `network.py` | Two architectures (`KalahNet` plain MLP, `ResidualKalahNet`) sharing the same dual-head interface: `forward(x)` returns `(policy_logits, value)`. Also contains `NeuralEvaluator`, `save_checkpoint`, and `load_checkpoint`. |
| `mcts.py` | Single-leaf MCTS with PUCT or UCT selection. `SearchNode` holds per-node stats. `MCTS.search()` returns a `SearchResult` with visit counts and the derived policy. |
| `batched_mcts.py` | `BatchedMCTS` subclass that collects multiple leaves before calling the evaluator, amortising GPU overhead. Requires a `BatchEvaluator` (i.e. `NeuralEvaluator`). |
| `agents.py` | High-level `Agent` protocol plus concrete agents: `RandomAgent`, `GreedyAgent`, `MinimaxAgent` (alpha-beta), `NoisyAgent` (wrapper), `MCTSAgent` (wraps MCTS + evaluator). |
| `train.py` | `TrainConfig` (frozen hyperparameters), `ReplayBuffer`, `self_play_game`, and `train_step` (one gradient update, policy cross-entropy + value MSE loss). |
| `evaluate.py` | `play_game`, `random_opening`, `choose_stones`, `choose_opening_plies`. |

### Training loop (in `scripts/train.py`)

1. Initialise (or resume) model, optimizer, and replay buffer.
2. Repeat until `--games` total have been played:
   a. Run `self_play_game` for `games_per_iteration` games → add samples to `ReplayBuffer`.
   b. Run `train_step` for `epochs` gradient steps on random batches from the buffer.
3. Save checkpoint on `Ctrl+C` or every `--checkpoint-every` games.

### Board encoding

`GameState.board` is a flat tuple of length `2*pits + 2`:
- indices `0..pits-1`: player 0 pits
- index `pits`: player 0 store
- indices `pits+1..2*pits`: player 1 pits
- index `2*pits+1`: player 1 store

Actions are always *local* pit numbers `0..pits-1` (not board indices). Both `GameState` and `FastGameState` share this convention.

### Value sign convention

All values are stored from the **current player's perspective** at each node. `_backup` in `mcts.py` negates the value whenever the player changes along the backup path.

### Checkpoint format

Saved by `network.save_checkpoint`: a dict with `model_state`, `pits`, `model_type`, `hidden_size`, `residual_blocks`, `encoding_version`, and `step`. A version mismatch on `encoding_version` triggers a warning in `load_checkpoint`.

## Comments

Every method should have a short comment (or docstring) that says what the method is for — even when the name makes the purpose obvious to an expert. The audience for this codebase includes learners who are reading the code alongside the tutorials, so a brief one-line description at the top of each method is always welcome.

In addition, add an inline comment whenever the *why* behind a line is non-obvious: hidden constraints, non-trivial arithmetic, or behaviour that would surprise a reader unfamiliar with the domain.
