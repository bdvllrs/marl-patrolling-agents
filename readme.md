# MARL for patrolling agents

We provide here an environment for a predator/prey game. We explore two methods: a simple DQN architecture as well as a true Multi-Agent algorithm architecture using a Policy Gradient approach: Multi-Agent Deep Deterministic Policy Gradient (Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems (pp. 6379-6390)).

## Some results
After 1400 episodes of training.

DDQN 2vs2 | MADDPG 2vs2 | DDQN 2v1 Magic Switch
:---------:|:----------:|:-----------:
![](gifs/dqn-2v2.gif "DDQN 2vs2") | ![](gifs/maddpg_2v2.gif "MADDPG 2vs2") | ![](gifs/switch-dqn-2v1.gif "DDQN switch 2vs2")

## Environment
Blue dots represent preys and orange dots are predators.

### Action space
The action space is discrete.
Every agent can do one of `none`, `left`, `right`, `top`, `bottom`.

### State space
The state is perfectly known by all the agents.

The state is the 3D coordinates (x, y, z) for every agent.

