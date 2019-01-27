import numpy as np
import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def compute_discounted_return(gamma, rewards):
    """
    Args:
        gamma: discount factor.
        rewards: List of reward for the episode

    Returns: $\Sum_{t=0}^T \gamma^t r_t$

    """
    discounted_return = 0
    discount = 1
    for reward in rewards:
        discounted_return += reward * discount
        discount *= gamma
    return discounted_return


def get_enemy_positions(agent_index: int, agents: List, positions: List[float]) -> List[Tuple[float, float]]:
    """
    Returns the list of agents enemy positions
    Args:
        positions:
        agent_index:
        agents:
    Returns:
    """
    for k in range(len(agents)):
        if agents[k].type != agents[agent_index].type:
            x, y, z = positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]
            yield (x, y, z, k)


def to_onehot(values, max):
    """
    Args:
        values:  dim batch_size
        max:
    Returns:

    """
    return torch.zeros(values.size(0), max).type_as(values).scatter_(1, values, 1).to(torch.float)


def np_to_onehot(values, max):
    onehot = np.zeros((len(values), max))
    onehot[np.arange(len(values)), values] = 1
    return onehot


def train(env, agents, memory, metrics, action_dim, config, agents_type="dqn"):
    all_rewards = []
    all_states = []
    all_next_states = []
    all_actions = []
    all_types = []

    states, types = env.reset()
    terminal = False
    step_k = 0
    while not terminal:
        actions = []
        for i in range(len(agents)):
            action = agents[i].draw_action(states[i])
            actions.append(action)
        all_types.append(types)
        next_states, rewards, terminal, n_collisions, types = env.step(states, actions)
        all_rewards.append(rewards)
        all_states.append(states)
        all_next_states.append(next_states)

        all_actions.append(actions)
        step_k += 1

        if agents_type == "maddpg":
            actions = np_to_onehot(actions, action_dim)

        memory.add(states, next_states, actions, rewards)

        # Learning step
        # Get batch for learning (batch_size x n_agents x dim)
        batch = memory.get_batch(config.learning.batch_size, shuffle=config.replay_memory.shuffle)

        if batch is not None:
            for k in range(len(agents)):
                if agents_type == 'maddpg':
                    loss_critic, loss_actor = agents[k].learn(batch)
                    metrics[k].add_loss(loss_critic)
                    metrics[k].add_loss_actor(loss_actor)
                else:
                    loss = agents[k].learn((batch[0][:, k], batch[1][:, k], batch[2][:, k], batch[3][:, k]))
                    metrics[k].add_loss(loss)

        states = next_states

    return all_states, all_next_states, all_rewards, all_actions, all_types


def test(env, agents, collision_metric, metrics, config):
    all_states = []
    all_rewards = []
    all_types = []
    states, types = env.reset(test=True)
    terminal = False
    while not terminal:
        actions = []
        for i in range(len(agents)):
            action = agents[i].draw_action(states[i], no_exploration=True)
            actions.append(action)
        all_types.append(types)
        next_states, rewards, terminal, n_collisions, types = env.step(states, actions)
        collision_metric.add_collision_count(n_collisions)
        all_rewards.append(rewards)
        all_states.append(states)

        states = next_states

    # Compute discounted return of the episode
    for k in range(len(agents)):
        reward = [all_rewards[i][k] for i in range(len(all_rewards))]
        discounted_return = compute_discounted_return(config.agents.gamma, reward)
        metrics[k].add_return(discounted_return)
    return all_states, all_rewards, all_types


def make_gif(env, fig, ax, states, rewards, types):
    def update(k):
        ax.cla()
        env.plot(states[k], types[k], rewards[k], ax)
        plt.draw()
        plt.pause(0.001)

    return FuncAnimation(fig, update, frames=np.arange(0, len(states)), interval=80)
