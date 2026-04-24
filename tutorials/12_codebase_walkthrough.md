# 12. Reading the Whole Codebase

## Goal

This final lesson gives you a map of the code. The project is small enough that
you can read all of it, but it helps to know the order.

The whole AlphaZero loop is:

$$
\text{GameState}
\rightarrow
\text{MCTS}
\rightarrow
\pi
\rightarrow
\text{ReplayBuffer}
\rightarrow
\mathcal{L}(\theta)
\rightarrow
f_{\theta'}
\rightarrow
\text{MCTS again}.
$$

## One Move Through The System

Start with a state:

```python
state = GameState.new_game()
```

Search it:

```python
result = mcts.search(state, evaluator)
```

Choose an action:

```python
action = result.select_action(temperature=temperature, rng=rng)
```

Apply it:

```python
state = state.apply(action)
```

Store the training target:

```python
trajectory.append((state, tuple(result.policy), state.current_player))
```

Mathematically, this creates:

$$
(s_t, \pi_t, p_t),
$$

where $p_t$ is the player who will later receive the final outcome target.

## Reading Order

Read the files in this order:

1. `src/kalah_zero/game.py`
2. `tests/test_game.py`
3. `src/kalah_zero/agents.py`
4. `src/kalah_zero/mcts.py`
5. `src/kalah_zero/encoding.py`
6. `src/kalah_zero/network.py`
7. `src/kalah_zero/train.py`
8. `src/kalah_zero/evaluate.py`

This order follows the dependency graph:

$$
\text{rules}
\rightarrow
\text{baselines}
\rightarrow
\text{search}
\rightarrow
\text{network}
\rightarrow
\text{training}
\rightarrow
\text{evaluation}.
$$

## The Core Interfaces

The important public methods are:

```python
state.legal_actions()
state.apply(action)
state.reward_for_player(player)
mcts.search(state, evaluator)
agent.select_action(state)
train_step(model, optimizer, batch)
```

If you understand those six calls, you understand the spine of the project.

## Mental Model

AlphaZero alternates between two operations:

$$
\text{policy improvement:}
\qquad
p_\theta(\cdot \mid s)
\mapsto
\pi(\cdot \mid s),
$$

$$
\text{policy evaluation and imitation:}
\qquad
\theta
\mapsto
\theta'
\text{ by minimizing }
\mathcal{L}(\theta).
$$

The beautiful part is that both operations use the same game engine.

## Practice

Do this walkthrough:

```bash
python -m pytest
python scripts/inspect_position.py --simulations 50 --tree-depth 1
python scripts/train.py --games 1 --simulations 5 --epochs 1
```

Then open the files above and trace one state from construction to training
sample. Write down every function it passes through.

