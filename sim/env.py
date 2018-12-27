import numpy as np
import matplotlib.pyplot as plt
from utils import choice, possible_directions, distance_enemies_around, position_from_direction
from utils.rewards import sparse_reward, full_reward

__all__ = ["World"]


class World:
    """
    Environment. An agent takes 1 pixel il the board
    """

    agent_radius = 1
    noise = 0.2
    max_iterations = None
    current_iter = 0

    def __init__(self, width, height, reward_type='full'):
        """
        Args:
            width: width of the board
            height: height of board
            reward_type (str): (full|sparse). Use default reward functions in utils.rewards. Defaults to full
        """
        assert reward_type in ['full', 'sparse'], "Unknown reward type."

        self.width = width
        self.height = height
        self.agents = []
        self.initial_positions = []
        self.reward_function = sparse_reward if reward_type == 'sparse' else full_reward

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
            self.agents[k].set_reward(reward)
        return rewards

    def step(self):
        self.current_iter += 1
        terminal_state = False if self.max_iterations is None else self.current_iter >= self.max_iterations
        actions = []
        positions = []
        for agent in self.agents:
            obs = []
            for other_agent in self.agents:
                if agent.type == 'target' and other_agent.type == 'officer':
                    num_officers_around = len(distance_enemies_around(agent, self.agents, max_distance=1))
                    if num_officers_around >= 2:
                        terminal_state = True
            action = agent.draw_action(obs)
            actions.append(action)
            if np.random.rand() < self.noise:
                # We select a position at random and not the one selected
                action = choice(possible_directions(agent.limit_board, agent.position))
            next_position = position_from_direction(agent.position, action)
            agent.set_position(next_position)
            positions.append(agent.position)
            agent.add_action_to_history(action)

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
