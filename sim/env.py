import numpy as np
import matplotlib.pyplot as plt
from utils import choice, possible_directions, get_distance_between
from utils.rewards import sparse_reward

__all__ = ["World"]


class World:
    """
    Environment. An agent takes 1 pixel il the board
    """

    agent_radius = 1
    noise = 0.2
    max_iterations = None
    current_iter = 0

    def __init__(self, width, height, reward_function=None):
        """
        Args:
            width: width of the board
            height: height of board
            reward_function: reward function. Default sparse reward as defined by utils.rewards.sparse_reward.
                The reward function must take a list of agents as input and return a list of reward (one for every
                agent)
        """
        self.width = width
        self.height = height
        self.agents = []
        self.initial_positions = []
        self.first_draw = True
        self.reward_function = reward_function if not None else sparse_reward

    def random_position(self):
        return np.random.randint(0, self.width), np.random.randint(0, self.height)

    def add_agent(self, agent, position=None):
        self.initial_positions.append(position)
        agent.set_size_board(self.width, self.height)
        self.agents.append(agent)

    def reset(self):
        self.current_iter = 0
        positions = []
        for k, agent in enumerate(self.agents):
            agent.reset()  # resets agents
            # Resets original positions (either fixed or random)
            position = self.initial_positions[k]
            position = self.random_position() if position is None else position
            positions.append(position)
            agent.set_position(position)
        return positions

    def give_rewards(self):
        """
        Gives reward to all agents
        """
        rewards = self.reward_function(self.agents)
        for k, reward in enumerate(rewards):
            self.agents[k].get_reward(reward)
        return rewards

    def step(self):
        self.current_iter += 1
        terminal_state = False if self.max_iterations is None else self.current_iter >= self.max_iterations
        actions = []
        positions = []
        for agent in self.agents:
            obs = []
            nb_patrol_around_target = 0
            for other_agent in self.agents:
                if agent.type == 'target' and other_agent.type == 'officer':
                    if other_agent.position in agent.view_area:
                        obs.append(other_agent)
                    if get_distance_between(agent.limit_board, agent.position, other_agent.position) == 1:
                        nb_patrol_around_target += 1
            if nb_patrol_around_target >= 2:
                terminal_state = True
            action = agent.draw_action(obs)
            actions.append(action)
            if np.random.rand() < self.noise:
                # We select a position at random and not the one selected
                action = choice(possible_directions(agent.limit_board, agent.position))
            agent.set_position(action)
            positions.append(agent.position)

        # Give rewards
        rewards = self.give_rewards()
        return positions, actions, rewards, terminal_state

    def draw_board(self):
        plt.ylim(bottom=0, top=self.height)
        plt.xlim(left=0, right=self.width)
        plt.grid(True)
        for agent in self.agents:
            agent.plot(self.agent_radius)
        plt.show()
