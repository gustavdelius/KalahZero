---
title: "AI Game Learning in Plain Language"
---

This page is for readers who are curious about AI game playing but do not yet
know the vocabulary. You can read it before the technical tutorials, or keep it
open as a companion while you work through them.

KalahZero teaches one central idea: a computer player can become stronger by
combining clear game rules, careful lookahead, and a learned sense of which
positions are promising.

## The Game Is the World

An AI player needs a world it can understand. In KalahZero, that world is the
board game Kalah.

The code describes:

- whose turn it is,
- how many stones are in each pit,
- which moves are legal,
- what happens after a move,
- when the game is over,
- who won.

This may sound humble, but it is the foundation of everything else. Before a
computer can learn good play, it needs a reliable way to ask, "What happens if I
do this?"

## An Agent Is a Player

In these tutorials, an **agent** is simply something that chooses a move.

A very simple agent might choose a legal move at random. A slightly smarter one
might choose the move that immediately captures the most stones. A stronger one
might look several turns ahead and ask which future positions seem best.

The word "agent" can sound grand, but here it means "the part of the program
that decides what to do next."

## Search Means Looking Ahead

When people play board games, they often imagine possible futures:

1. If I move here, my opponent might move there.
2. Then I could respond this way.
3. That future board looks good, or bad, for me.

That is the basic idea behind **search**. The program explores possible move
sequences before choosing its real move.

The technical tutorials introduce Monte Carlo Tree Search, often shortened to
MCTS. The plain-language version is:

- build a branching tree of possible futures,
- spend more attention on branches that look useful,
- use the results to choose a move.

Search does not make the player magically intelligent. It gives the player a
disciplined way to think before acting.

## A Neural Network Learns Patterns

A **neural network** is a function with many adjustable numbers inside it. At
first, its guesses are not useful. During training, those numbers are adjusted
so that its guesses become better.

In KalahZero, the neural network learns two things:

- which moves look promising from a given board position,
- whether that position looks good for the current player.

You can think of it as a developing sense of taste. It does not replace the
rules of the game, and it does not replace search. It helps search focus on
better-looking possibilities sooner.

## Self-Play Means Practicing Against Yourself

AlphaZero-style systems do not need a library of expert human games. They can
generate practice games by playing against themselves.

In each self-play game:

1. the current agent uses search to choose moves,
2. the finished game reveals who eventually won,
3. the training code asks the neural network to better predict those choices
   and outcomes next time.

Over many games, the network and search procedure can reinforce each other. The
network guides search; search produces better training examples; the improved
network guides the next round of search.

## Why Evaluation Matters

Training can produce a newer agent, but newer does not automatically mean
better. KalahZero includes evaluation tools so different agents can play matches
against each other.

That lets you ask practical questions:

- Does the trained agent beat the random agent?
- Does it beat the greedy agent?
- Does deeper search help?
- Did a change make play stronger, weaker, or just different?

This is one of the most important habits in machine learning: check behavior
with experiments, not vibes.

## Where to Go Next

If you want the gentlest next step, play a few games in
[the browser app](../web/index.qmd). Notice when the computer makes an obvious
move and when it surprises you.

If you are ready to read the technical path, continue with
[Kalah Rules and State Representation](01_kalah_rules.md). That chapter turns
the game into precise code, which makes every later idea possible.

If you mostly want the big picture, skim the tutorial titles in order. They
trace the same story you just read:

rules, simple agents, search, neural networks, self-play, and evaluation.
