---
title: "AI Game Learning in Plain Language"
---

This page is for readers who are curious about AI game playing but do not yet
know the vocabulary. You can read it before the technical tutorials, or keep it
open as a companion while you work through them.

KalahZero teaches one central idea: a computer player can become stronger by
combining clear game rules, careful lookahead, and a learned sense of which
positions are promising.

That combination will sound familiar. It is also how *you* play board games.

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

When you sit down to play Kalah, you do exactly this — even if you never call it
"search." You mentally pick up stones from one pit, scatter them around the
board, notice whether the last stone falls into your store for a bonus turn, and
decide whether the result looks promising. Then you imagine your opponent's reply
and do it again. That internal conversation *is* lookahead, and it is the same
thing the computer does.

The main difference is stamina. You might trace two or three moves ahead before
it becomes too hard to track. A computer can trace dozens of branches in the
time it takes you to blink, and it never loses count. But the underlying idea —
play out possibilities in your head, then commit to the move that leads to the
best-looking future — is shared.

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

Think of it like a student learning chess by playing thousands of games. After
enough practice, the student develops a feeling — without consciously working
it out — that "boards with both rooks active and a strong centre tend to be
good." They cannot always explain why a position feels winning; they just
*know*. A neural network builds a similar kind of pattern recognition, except
through adjustment of numbers rather than through conscious reflection.

### What the Network Sees

Before the network can do anything, the board state must be turned into numbers.
In KalahZero that means listing, in a fixed order:

- how many stones are in each of your own pits,
- how many stones are in each of your opponent's pits,
- how many stones are in each player's store,
- a constant bias term (always 1).

That flat list of numbers is the network's only window onto the world. It never
sees a picture of the board or reads a description. Everything it knows must be
inferred from those numbers. Surprisingly, that is enough.

### Two Answers from One Glance

In KalahZero, the neural network produces two outputs at once from one
reading of the board position.

**The value** is a single number between −1 and +1. It answers the question
"How good is this position for the player whose turn it is right now?" A value
near +1 means the network thinks you are likely to win from here; near −1
means you are likely to lose; near 0 means the outcome looks uncertain.

**The policy** is a list of numbers, one for each legal move. It answers the
question "Which moves look most worth exploring?" A move with a high policy
score is one the network has learned to favour from positions like this one. You
can think of it as the network pointing and saying "I'd try *that* move first."

Together, value and policy are a learned intuition: *this position looks good,
and here is roughly what I'd do about it.*

You produce the same two things when you glance at a Kalah board. Before you
have carefully worked out any sequence of moves, you already have a rough feeling
about whether the board looks good for you — stones spread well, store ahead,
opponent's pits thinning out — and a rough sense of which pit or two seems most
worth picking up. You probably do not notice these as two separate thoughts;
they arrive together in a single impression. But they are there, and they are
exactly the value and the policy that the network is learning to mimic.

### How the Intuition Develops

At the very start of training the network is initialised with random numbers.
Its value guesses are meaningless and its policy suggestions are close to
random. It is the equivalent of a beginner who has just learned the rules.

As training progresses the network's numbers are nudged — thousands of times —
to bring its outputs closer to what actually happened in real games. If the
network said a position looked promising and the game was then lost, its numbers
are adjusted so it becomes slightly less optimistic about similar positions in
the future. If it correctly identified a winning move, the numbers that led to
that suggestion are reinforced.

Over many games and many such adjustments, the network begins to notice real
patterns: "positions where many stones sit in my larger pits and my store is
ahead tend to be winning," or "that pit with a single stone can often be
exploited." It never states these patterns in words; they are captured implicitly
in thousands of small numbers. But the effect is a genuine sense of taste — the
ability to look at a board and quickly feel which directions are worth pursuing.

### Intuition Helps Search, but Does Not Replace It

The network's intuition is fast but fallible. It can be wrong about positions
it has never seen anything like before. Search — looking ahead step by step —
is slower but more reliable in the positions it actually reaches.

AlphaZero's key insight is that the two complement each other perfectly.
The network makes search efficient by telling it where to look; search makes
the network better by generating high-quality training examples that go beyond
raw game outcomes.

## Self-Play Means Practicing Against Yourself

AlphaZero-style systems do not need a library of expert human games. They can
generate practice games by playing against themselves.

In each self-play game:

1. the current agent uses search to choose moves,
2. the finished game reveals who eventually won,
3. the training code asks the neural network to better predict those choices
   and outcomes next time.

The feedback goes in two directions. From the game's outcome, the network learns
to adjust its *value* estimates — positions that led to a win should have looked
more promising, and vice versa. From the move choices made during search, the
network learns to adjust its *policy* — moves that the careful search preferred
should score higher next time, even before any full game is played.

Over many games, the network and search procedure reinforce each other. The
network guides search toward promising branches; search produces better training
examples; the improved network guides the next round of search still more
reliably. This loop is why the system can reach strong play starting from
nothing but the rules of the game.

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
[the browser app](../web/index.qmd). As you play, pay attention to your own
thinking: notice the moment a move *feels* right before you have checked it
fully — that is your intuition speaking. Notice when you deliberately trace out
a sequence of moves to verify it — that is your lookahead. The computer is doing
both of those things too, just faster and without getting bored.

If you are ready to read the technical path, continue with
[Kalah Rules and State Representation](01_kalah_rules.md). That chapter turns
the game into precise code, which makes every later idea possible.

If you mostly want the big picture, skim the tutorial titles in order. They
trace the same story you just read:

rules, simple agents, search, neural networks, self-play, and evaluation.
