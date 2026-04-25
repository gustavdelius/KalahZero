# KalahZero

KalahZero is a teaching repo for learning AlphaZero by building a small
AlphaZero-style agent for Kalah.

The project is intentionally compact:

- `tutorials/` explains the math and code in order.
- `_quarto.yml` turns those tutorials into a Quarto website with MathJax.
- `src/kalah_zero/` contains the implementation.
- `tests/` protects the game rules and search behavior.
- `scripts/` gives you command-line entry points for play, training, and
  inspection.

Default game: 6 pits per player, 4 stones per pit, skip the opponent store,
extra move when the last stone lands in your store or when you make a capture,
capture from the opposite pit when the last stone lands in an empty own pit,
and sweep remaining stones when either side becomes empty.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Run a tiny self-play/training loop:

```bash
python scripts/train.py --games 4 --simulations 25 --epochs 2
```

Longer training runs are resumable. The trainer saves periodic checkpoints and
also saves when interrupted with `Ctrl+C`:

```bash
python scripts/train.py --games 300 --simulations 75 --epochs 1 --output checkpoints/overnight.pt
python scripts/train.py --resume checkpoints/overnight.pt
```

Use `--checkpoint-every N` to control periodic saves. On resume, `--games`
means the total number of games you want completed, not the number of additional
games. To speed neural self-play on a laptop CPU, opt into batched MCTS leaf
evaluation:

```bash
python scripts/train.py --games 300 --simulations 150 --epochs 1 --batched-mcts --eval-batch-size 32
```

For higher simulation counts, build the optional C++ game engine and use it
together with batched MCTS:

```bash
python setup.py build_ext --inplace
python scripts/evaluate.py --checkpoint-a checkpoints/overnight.pt --agent-b minimax \
  --simulations 400 --batched-mcts --eval-batch-size 8 --fast-game
```

The same opt-in works for checkpoint evaluation:

```bash
python scripts/evaluate.py --checkpoint-a checkpoints/overnight.pt --agent-b minimax --simulations 200 --batched-mcts
```

Noisy baselines are available for human-like mistake rates:

```bash
python scripts/evaluate.py --checkpoint-a checkpoints/overnight.pt --agent-b noisy-minimax --noise-prob 0.1
```

To train on less opening-specific positions, start each self-play game after a
few random legal moves:

```bash
python scripts/train.py --resume checkpoints/overnight.pt --games 2500 --opening-plies 4
```

Or sample a different opening length each game:

```bash
python scripts/train.py --resume checkpoints/overnight.pt --games 3000 --opening-plies-min 0 --opening-plies-max 8
```

To compare network capacity experiments cleanly, start separate checkpoints:

```bash
# depth only: residual MLP, same hidden width as the baseline
python scripts/train.py --model-type residual --hidden-size 128 --residual-blocks 3 --output checkpoints/residual_depth.pt

# width only: plain MLP, wider hidden representation
python scripts/train.py --model-type mlp --hidden-size 256 --output checkpoints/mlp_width.pt

# depth plus width
python scripts/train.py --model-type residual --hidden-size 256 --residual-blocks 3 --output checkpoints/residual_width.pt
```

Play against a baseline:

```bash
python scripts/play_cli.py --agent greedy
```

Open the graphical browser board:

```bash
quarto preview
```

Then open the **Play** page in the navbar. The **Trained agent** opponent runs
entirely in the browser using the exported `web/residual_depth.json` weights and
a Web Worker, so it does not need a local server.

To refresh the browser weights from a checkpoint:

```bash
python scripts/export_browser_agent.py checkpoints/residual_depth.pt web/residual_depth.json
```

Read the tutorials in order from `tutorials/01_kalah_rules.md`.

## Tutorial Website

Render the Quarto site locally:

```bash
quarto render
quarto preview
```

The GitHub Actions workflow in `.github/workflows/quarto-pages.yml` renders
the site and deploys `_site` to GitHub Pages. In the repository settings, set
Pages to use **GitHub Actions** as its source.
