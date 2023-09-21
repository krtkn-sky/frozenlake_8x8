# Frozen Lake Q-Learning

This code implements Q-Learning to solve the Frozen Lake 8x8 environment in OpenAI Gym.

## Overview

The Frozen Lake environment is a grid world where the agent must navigate from a starting point to a goal while avoiding holes. Q-Learning is used to train the agent to make optimal decisions based on rewards received for each action.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- Gym
- NumPy
- Matplotlib

You can install these libraries using `pip`:

```bash
pip install gym numpy matplotlib
```
## Usage

To run the code, use the run function in the main script. You can configure the following parameters:

-episodes: The number of training episodes.
-is_training: Set to True to train the agent, False to load a pre-trained model.
-render: Set to True to visualize the training process.

### Example usage:

```bash
if __name__ == '__main__':
    run(episodes=1000, is_training=True, render=True)```

