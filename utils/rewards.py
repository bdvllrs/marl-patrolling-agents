from utils import get_distance_between


def sparse_reward(agents):
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
        if agent.type == 'target':
            num_officier_around = 0
            for other_agent in agents:
                if (agent.type == 'officer' and
                        get_distance_between(agent.limit_board, agent.position, other_agent.position) == 1):
                    num_officier_around += 1
            if num_officier_around >= 2:
                officer_won = True
    # Define reward for all agents
    for agent in agents:
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
