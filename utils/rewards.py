from utils import distance_enemies_around


def sparse_reward(agents, t):
    """
    Gives a sparse reward at the end:
        - 0 for officers if target is alone and 1 for target
        - 1 for officer if target is surrounded and 0 for target
    Args:
        agents:
    Returns: list of rewards for every agent
    """
    rewards = []
    officer_won = False
    # Check who won
    for agent in agents:
        if agent.type == "target":
            num_officers_around = len(distance_enemies_around(agent, agents, max_distance=1))
            if num_officers_around >= 2:
                officer_won = True
    # Define reward for all agents
    for agent in agents:
        new_x, new_y = agent.position
        if new_x >= agent.limit_board[0] or new_y >= agent.limit_board[1] or new_x < 0 or new_y < 0:
            rewards.append(0)
        else:
            if agent.type == 'officer':
                if officer_won:
                    rewards.append(1)
                else:
                    rewards.append(0)
            else:
                if officer_won:
                    rewards.append(0)
                else:
                    rewards.append(1)
    return rewards


def dist_reward(agents, t):
    rewards = []
    for agent in agents:
        x, y = agent.position
        rewards.append(5 - (x - 10) ** 2 - (y - 10) ** 2)
    return rewards


def full_reward(agents, t):
    """
    Gives a reward at each step:
    The reward depends on the distance of the agent to its "enemy".
    If an officer is close to the target, it wins some reward.
    If the target cannot see any officer, it wins some reward.
    Args:
        agents:
    Returns: list of rewards for every agent
    """
    rewards = []
    num_officers, num_targets = 0, 0
    for agent in agents:
        if agent.type == "target":
            num_targets += 1
        if agent.type == 'officer':
            num_officers += 1
    for agent in agents:
        if agent.type == "officer" and len(distance_enemies_around(agent, agents, max_distance=1)) > 0:
            rewards.append(100)
        else:
            new_x, new_y = agent.position
            if new_x >= agent.limit_board[0] or new_y >= agent.limit_board[1] or new_x < 0 or new_y < 0:
                rewards.append(-50)
            else:
                distances = distance_enemies_around(agent, agents)
                if agent.type == "target":
                    if len(distances) == 0:
                        reward = agent.view_radius * num_officers
                    else:
                        reward = sum(distances)
                else:
                    if len(distances) == 0:
                        reward = 0
                    else:
                        reward = 0.001*(agent.view_radius * num_targets - sum(distances))
                # reward /= 2 ** t
                rewards.append(reward)
    return rewards


def compute_discounted_return(rewards, gamma):
    """
    Compute discounted return of an episode
    Args:
        rewards: rewrads of the episode
        gamma: discount factor
    """
    discounted_return = 0
    discount = 1
    for reward in rewards:
        discounted_return += discount * reward
        discount *= gamma
    return discounted_return
