from typing import List
from sim.agents import Agent
import numpy as np
from utils.config import Config

config = Config('./builds')




def reward_full(observations, agents: List[Agent], t):
    """
    give all the rewards
    :param observations: all board
    :param agents: all agents list
    :param t: time
    :return: liste of all reward
    """
    all_winners = []
    all_rewards = []
    perdu = False
    for idx, agent in enumerate(agents):
        if agent.type == "prey":
            (rw, win) = reward_full_agent(observations, idx, agents, t)
            if win:
                perdu = True
        if agent.type == "predator":
            (rw, win) = reward_full_agent(observations, idx, agents, t)
        all_winners.append(win)
        all_rewards.append(rw)
    if perdu:
        for idx, agent in enumerate(agents):
            if agent.type == "prey":
                if all_winners[idx]:
                    all_rewards[idx] += 1
                else:
                    all_rewards[idx] += 0.8
            else:
                all_rewards[idx] += -1
    return all_rewards

def reward_full_agent(observations, agent_index, agents: List[Agent], t):
    """
    reward for 1 action and 1 agent at time t
    :param observations:
    :param agent_index:
    :param agents:
    :param t:
    :return: reward value
    """
    x = observations[2 * agent_index]
    y = observations[2 * agent_index + 1]
    if agents[agent_index].type == "predator":
        for idx, agent in enumerate(agents):
            if agent.type == "prey":
                rew, dis = distance_reward(x, y, observations, idx)
                rew_d = 1 - rew
                if dis <= config.env.board_size:
                    winner = True
                else:
                    winner = False
    else:
        for idx, agent in enumerate(agents):
            if agent.type == "predator":
                rew, dis = distance_reward(x, y, observations, idx)
                rew_d = rew
                winner = False

    return rew_d, winner

def distance_reward(x, y, observations, idx):
    x1 = observations[2 * idx]
    y1 = observations[2 * idx + 1]
    distance = np.linalg.norm((x, y) - (x1, y1))
    rw = config.reward.coef_distance_reward*distance
    return (rw, distance)
