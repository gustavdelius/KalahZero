# Tutorial Sequence

These tutorials teach AlphaZero by building it for Kalah. Read them in order:
each lesson introduces one mathematical idea, connects it to code, and ends
with a small practice task.

## How To Study

For each tutorial:

1. Read the goal and the mathematical object being introduced.
2. Copy the code excerpt into your editor and find the full implementation.
3. Run the suggested command.
4. Change one small thing and predict what should happen before running again.

The central loop is:

\[
\text{rules} \rightarrow \text{search} \rightarrow \text{targets} \rightarrow
\text{neural network} \rightarrow \text{better search}.
\]

## Lessons

1. [Kalah Rules and State Representation](01_kalah_rules.md)
2. [Perfect-Information Games](02_perfect_information_games.md)
3. [Baseline Agents](03_baseline_agents.md)
4. [Monte Carlo Tree Search](04_monte_carlo_tree_search.md)
5. [From UCT to AlphaZero PUCT](05_puct.md)
6. [Neural Policy and Value Networks](06_neural_policy_value_networks.md)
7. [Self-Play Training Loop](07_self_play_training.md)
8. [AlphaZero Loss Derivation](08_alphazero_loss.md)
9. [Evaluation and Elo-Style Comparisons](09_evaluation.md)
10. [Instrumentation and Debugging](10_instrumentation.md)
11. [Scaling Strength](11_scaling_strength.md)
12. [Reading the Whole Codebase](12_codebase_walkthrough.md)

