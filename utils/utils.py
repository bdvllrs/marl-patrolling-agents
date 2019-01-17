import torch
from typing import List, Tuple


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
            yield (x, y, z)


def to_onehot(values, max):
    """
    Args:
        values:  dim batch_size
        max:
    Returns:

    """
    return torch.zeros(values.size(0), max).scatter_(1, values, 1).to(torch.float)
