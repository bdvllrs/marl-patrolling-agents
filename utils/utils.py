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
            x, y = positions[2 * k], positions[2 * k + 1]
            yield (x, y)
