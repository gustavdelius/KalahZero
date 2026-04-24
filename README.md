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
extra move when the last stone lands in your store, capture from the opposite
pit when the last stone lands in an empty own pit, and sweep remaining stones
when either side becomes empty.

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

Play against a baseline:

```bash
python scripts/play_cli.py --agent greedy
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
