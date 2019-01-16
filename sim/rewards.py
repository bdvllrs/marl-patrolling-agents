from typing import List, Tuple
from sim.agents import Agent
import numpy as np
from utils.config import Config
from utils import get_enemy_positions

config = Config('./config')


def reward_full(observations, agents: List[Agent], border_positions, t):
    """
    give all the rewards
    :param observations: all board
    :param agents: all agents list
    :param t: time
    :return: liste of all reward
    """
    all_winners = []
    all_rewards = []
    predator_lost = True
    for idx, agent in enumerate(agents):
        rw, is_winner = get_reward_agent(observations, idx, agents, t)
        if agent.type == "predator" and is_winner:
            predator_lost = False

        all_winners.append(is_winner)
        all_rewards.append(rw)
    if config.reward.share_wins_among_predators or config.reward.hot_walls:
        for idx, agent in enumerate(agents):
            # if predator_lost:
            #     if agent.type == "prey":
            #         if not all_winners[idx]:
            #             all_rewards[idx] = -1
            # else:
            if config.reward.share_wins_among_predators and not predator_lost:
                # Give incentive to other predators if one wins
                if agent.type == "predator":
                    if not all_winners[idx]:
                        all_rewards[idx] = config.reward.reward_if_predators_win
            if config.reward.hot_walls:  # If agent touches the borders, gets a penalty
                x, y = observations[2 * idx], observations[2 * idx + 1]
                if x in border_positions or y in border_positions:
                    all_rewards[idx] = -1

    return all_rewards


def get_reward_agent(observations, agent_index, agents: List[Agent], t):
    """
    Reward for 1 action and 1 agent at time t
    :param observations:
    :param agent_index:
    :param agents:
    :param t:
    :return: reward value
    """
    x = observations[2 * agent_index]
    y = observations[2 * agent_index + 1]
    min_distance = None
    agent_reward = None
    winner = False
    agent_type = agents[agent_index].type

    # For all enemies, we compute the distance between actual agent and enemy agents
    for x_1, y_1 in get_enemy_positions(agent_index, agents, observations):
        reward, distance = distance_reward(agent_type, x, y, x_1, y_1)
        # If distance is smaller...
        if min_distance is None or distance < min_distance:
            min_distance, agent_reward = distance, reward

    # Determine if it's a winner
    if ((agent_type == "predator" and min_distance <= 1 / config.env.board_size) or
            (agent_type == "prey" and min_distance > 1 / config.env.board_size)):
        winner = True

    return agent_reward, winner


def distance_reward(agent_type, x, y, x1, y1) -> Tuple:
    """
    Returns the reward obtained according to the type and the position of two agents
    Args:
        agent_type: between prey and predator
        x: position x current agent
        y: position y current agent
        x1: position secondary agent
        y1: position secondary agent

    Returns: (reward, distance_between_agents)
    """
    assert agent_type in ['prey', 'predator'], "Agent type is not correct."
    return distance_reward_prey(x, y, x1, y1) if agent_type == "prey" else distance_reward_predator(x, y, x1, y1)


def distance_reward_prey(x, y, x1, y1):
    """
    Returns: Reward = $1 - \exp(-c \cdot d^2)$
    """
    distance = np.linalg.norm([x - x1, y - y1])
    rw = np.exp(-config.reward.coef_distance_reward * distance * distance)
    return 1 - rw, distance


def distance_reward_predator(x, y, x1, y1):
    """
    Returns: Reward = $\exp(-c \cdot d^2)$
    """
    distance = np.linalg.norm([x - x1, y - y1])
    rw = np.exp(-config.reward.coef_distance_reward * distance * distance)
    return rw, distance
