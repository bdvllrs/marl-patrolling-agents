# MARL for patrolling agents

## Examples after 1400 episodes of training

DDQN 2vs2 | MADDPG 2vs2 | DDQN 2v1 Magic Switch
:---------:|:----------:|:-----------:
![](gifs/dqn-2v2.gif "DDQN 2vs2") | ![](gifs/maddpg_2v2.gif "MADDPG 2vs2") | ![](gifs/switch-dqn-2v1.gif "DDQN switch 2vs2")

## Evironment

### Action space
Every agent can do one of `none`, `left`, `right`, `top`, `bottom`.

### State space
The state is complete and everything is known by the agents.

The state is the 3D coordinates (x, y, z) for every agent.

