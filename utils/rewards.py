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
    officer_won = False
    for agent in agents:
        if agent.type == "target":
            num_targets += 1
            num_officers_around = len(distance_enemies_around(agent, agents, max_distance=1))
            if num_officers_around >= 2:
                officer_won = True
        if agent.type == 'officer':
            num_officers += 1
    for agent in agents:
        if agent.type == "officer" and officer_won:
            rewards.append(1000/(2**t))
        else:
            new_x, new_y = agent.position
            if new_x >= agent.limit_board[0] or new_y >= agent.limit_board[1] or new_x < 0 or new_y < 0:
                rewards.append(-10000)
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
                        reward = agent.view_radius * num_targets - sum(distances)
                reward /= 2 ** t
                rewards.append(reward)
    return rewards
