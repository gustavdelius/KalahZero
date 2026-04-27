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

To train across different starting stone counts:

```bash
python scripts/train.py --resume checkpoints/residual_depth.pt --games 12000 --stones-min 4 --stones-max 6
```

To bias training toward harder starting stone counts, use relative weights:

```bash
python scripts/train.py --games 12000 --stone-weights 4:1,5:1,6:2
```

Evaluate exact 6-stone play with:

```bash
python scripts/evaluate.py --checkpoint-a checkpoints/residual_depth.pt --agent-b minimax --stones 6
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

## Overnight Coaching

After the fixed-count encoding change, start a fresh checkpoint for mixed
4/5/6-stone training. The overnight coach runs short train/evaluate cycles,
logs results, and keeps `best.pt` according to a score that favors winning with
fewer MCTS simulations:

```bash
python scripts/overnight_coach.py \
  --output-dir runs/fixed-scale-overnight \
  --blocks 8 \
  --block-games 500 \
  --train-simulations-schedule 250,250,200,200,150,150,100,100 \
  --epochs-schedule 1,1,1,1,1,1,2,2 \
  --eval-games 100
```

Each block evaluates exact 4-, 5-, and 6-stone play with random openings against
minimax at several simulation budgets. It also evaluates against noisy minimax
at a 10% mistake rate. Results are written to:

- `runs/fixed-scale-overnight/metrics.csv`
- `runs/fixed-scale-overnight/summary.json`
- `runs/fixed-scale-overnight/best.pt`

To keep the 4-, 5-, and 6-stone minimax results from drifting apart, enable
adaptive stone weighting:

```bash
python scripts/overnight_coach.py \
  --output-dir runs/fixed-scale-balanced \
  --balance-stone-weights \
  --eval-games 100
```

After every block, the coach updates the next training command's
`--stone-weights` so stone counts with lower minimax score rates are sampled
more often. By default, the update keeps 25% of the previous normalized weights
with `--stone-weight-smoothing 0.25`; set it to `0` for purely score-based
weights.

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
