import numpy as np
import random


def choice(l):
    """
    Returns a random element from the list
    """
    item = np.random.randint(0, len(l))
    return l[item]


def possible_directions(limit_board, position):
    """
    Gives the possible moves allowed for the agent
    Args:
        limit_board: limits of the board
        position: position of the agent
    Returns: list of possible positions
    """
    lim_x, lim_y = limit_board
    x, y = position
    possible_direction = ['none']
    if x > 0:
        possible_direction.append('left')
        if y > 0:
            possible_direction.append('bottom-left')
        if y < lim_y - 1:
            possible_direction.append('top-left')
    if y > 0:
        possible_direction.append('bottom')
    if x < lim_x - 1:
        possible_direction.append('right')
        if y > 0:
            possible_direction.append('bottom-right')
        if y < lim_y - 1:
            possible_direction.append('top-right')
    if y < lim_y - 1:
        possible_direction.append('top')
    return possible_direction


def position_from_direction(current_position, direction):
    """
    Gives position on the board from a direction
    Args:
        current_position: Position of the agent (x_cur, y_cur)
        direction: one of the 8 possible directions (none, top, left, ...)
    Returns: couple of position (x_new, y_new)
    """
    assert direction in ['none', 'top', 'left', 'right', 'bottom', 'top-left', 'top-right', 'bottom-right',
                         'bottom-left'], "The direction is unknown."
    x, y = current_position
    if direction == 'top':
        return x, y + 1
    elif direction == 'bottom':
        return x, y - 1
    elif direction == 'left':
        return x - 1, y
    elif direction == 'right':
        return x + 1, y
    elif direction == 'top-left':
        return x - 1, y + 1
    elif direction == 'top-right':
        return x + 1, y + 1
    elif direction == 'bottom-left':
        return x - 1, y - 1
    elif direction == 'bottom-right':
        return x + 1, y - 1
    else:  # none
        return x, y


def get_distance_between(limit_board, position_1, position_2):
    """
    Returns the distance between the two positions
    Args:
        limit_board: limits of the board
        position_1: position 1
        position_2: position 2
    """
    positions = [position_from_direction(position_2, direction) for direction in
                 possible_directions(limit_board, position_2)]
    if position_1 in positions:
        return 1
    x, y = position_1
    x_2, y_2 = position_2
    x_new, y_new = position_1
    if x < x_2:
        if y < y_2:
            x_new, y_new = x + 1, y + 1
        elif y == y_2:
            x_new = x + 1
        else:
            x_new, y_new = x + 1, y - 1
    elif x == x_2:
        if y < y_2:
            y_new = y + 1
        elif y > y_2:
            y_new = y - 1
    else:
        if y < y_2:
            x_new, y_new = x - 1, y + 1
        elif y == y_2:
            x_new = x - 1
        else:
            x_new, y_new = x - 1, y - 1
    return 1 + get_distance_between(limit_board, (x_new, y_new), position_2)


def distance_enemies_around(agent, agents, max_distance=None):
    """
    Returns the distance to all enemies around one agent
    Args:
        agent: reference agent
        agents: other agents
        max_distance: max distance. Default agent's field of view
    Returns: distances
    """
    max_distance = agent.view_radius if max_distance is None else max_distance
    enemies_around = []
    for other_agent in agents:
        if ((agent.type == "target" and other_agent.type == 'officer') or
                (agent.type == "officer" and other_agent.type == 'target')):
            dist_agents = get_distance_between(agent.limit_board, agent.position, other_agent.position)
            if dist_agents <= max_distance:
                enemies_around.append(dist_agents)
    return enemies_around


def state_from_observation(agent, observation):
    """
    Get state board from observation.
    It is a board of the field of view of the agent where:
    - -1 is out of board
    - 0.5 is friend
    - 1 is enemy
    Args:
        agent: Agent receiving the observation
        observation:
    Returns: array of shape (2 * agent.view_radius + 1, 2 * agent.view_radius + 1)
    """
    side_length = 2 * agent.view_radius + 1
    board = [[-1 for i in range(side_length)] for j in range(side_length)]
    view_area = agent.view_area
    x_a, y_a = agent.position
    for (x, y) in view_area:
        i, j = int(x - x_a + agent.view_radius), int(y - y_a + agent.view_radius)
        board[i][j] = 0
    for obs in observation:
        x, y = obs.position
        i, j = int(x - x_a + agent.view_radius), int(y - y_a + agent.view_radius)
        if obs.type == agent.type:
            board[i][j] = 0.5
        else:
            board[i][j] = 1
    return np.array(board)


def sample_batch_history(agent, batch_size, memory=10000):
    """
    Samples the history of the given agent.
    Args:
        agent: agent to sample
        batch_size:
        memory: how far we can go back (max) in the history
    Returns: None or dictionary of keys "states", "next_states", "actions" and "rewards" and values a list of the values in
        the history. *states* corresponds to prev_position, *next_states* to position, *actions* to action and
        *rewards* to reward. position is set to None if it is the last position (terminal).
        Returns None if there are not enough elements in the history to fill a full batch.
    """
    history = list(filter(lambda x: "action" in x.keys(), agent.histories))[-memory:]
    if len(history) >= batch_size:
        batch = random.sample(history, batch_size)
        return {
            "states": list(map(lambda x: x["prev_position"], batch)),
            "next_states": list(map(lambda x: x["position"] if not x["terminal"] else None, batch)),
            "actions": list(map(lambda x: x["action"], batch)),
            "rewards": list(map(lambda x: x["reward"], batch)),
        }
    return None
