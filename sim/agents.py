import numpy as np
import matplotlib.pyplot as plt
from utils import choice, possible_directions, position_from_direction, get_distance_between

__all__ = ["Officer", "Target", "Headquarters"]


class Agent:
    type_id = None
    type = None
    position = None
    last_reward = None
    last_action = None
    color = "red"
    limit_board = None
    view_radius = 10  # manhattan distance

    def __init__(self, name=None):
        assert (self.type_id is not None and
                self.type is not None), "This agent does not have any type information."
        if name is None:
            name = self.type + " " + str(np.random.randint(0, 1000))
        self.name = name
        self.id_to_action = ['none', 'top', 'left', 'right', 'bottom', 'top-left', 'bottom-left', 'top-right',
                             'bottom-right']
        self.action_to_id = {a: i for i, a in enumerate(self.id_to_action)}
        self.histories = []  # Memory for all the episodes

    def reset(self):
        if len(self.histories) > 0:
            self.histories[-1]["terminal"] = True  # New history

    def set_position(self, position):
        self.histories.append({"position": position, "terminal": False})  # Keep history of all positions
        self.position = position
        if len(self.histories) > 1 and not self.histories[-2]["terminal"]:
            self.histories[-1]["prev_position"] = self.histories[-2]["position"]

    def add_action_to_history(self, action):
        self.histories[-1]['action'] = action
        self.last_action = action

    def set_reward(self, reward):
        """
        Sets reward for last action.
        Has to be called after a set_position
        Args:
            reward:
        """
        self.histories[-1]["reward"] = reward
        self.last_reward = reward

    @property
    def view_area(self):
        """
        Defines the points in the board the agent can see
        """
        positions = [self.position]
        self.view_area_from(self.position, self.view_radius, positions)
        return positions

    def view_area_from(self, position, hops, positions):
        """
        Defines the points in the board the agent can see from the given position
        Args:
            positions: list of seen positions
            position: original position
            hops: Number of allowed hops
        """
        if hops == 0:
            return
        directions = possible_directions(self.limit_board, position)
        for direction in directions:
            new_position = position_from_direction(self.position, direction)
            if new_position not in positions:
                positions.append(new_position)
                self.view_area_from(new_position, hops - 1, positions)
        return

    def set_size_board(self, width, height):
        self.limit_board = (width, height)

    def plot(self, radius):
        x, y = self.position
        circle = plt.Circle((x, y), radius=radius, color=self.color)
        plt.gcf().gca().add_artist(circle)

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other agents. List of agents.
        """
        return choice(possible_directions(self.limit_board, self.position))


class Officer(Agent):
    """
    No MARL for this one. Each of them learns individually
    """
    type_id = 0
    type = "officer"
    color = "blue"

    def __init__(self, name=None):
        super(Officer, self).__init__(name)

        # For thompson sampling
        self.S = np.zeros(len(self.id_to_action), dtype=float)
        self.F = np.zeros(len(self.id_to_action), dtype=float)

    def set_reward(self, reward):
        """
        Generalized Thompson Sampling. Inspired by Agrawal and Goyal, 2012.
        https://arxiv.org/pdf/1111.1797.pdf
        """
        bernoulli_trial = float(np.random.rand() < reward)  # Generalized version
        last_action = self.histories[-1]['action']
        self.S[self.action_to_id[last_action]] += bernoulli_trial
        self.F[self.action_to_id[last_action]] += 1 - bernoulli_trial
        super(Officer, self).set_reward(bernoulli_trial)

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other agents. List of agents.
        """
        theta = np.random.beta(self.S + 1, self.F + 1)
        draw = np.argmax(theta).item()
        return self.id_to_action[draw]


class Headquarters(Agent):
    """
    MARL - represents several Officers
    """
    type_id = 2
    type = "hq"
    color = "black"


class Target(Agent):
    type_id = 1
    type = "target"

    def distance_to_officers(self, obs, direction=None):
        position = self.position if direction is None else position_from_direction(self.position, direction)
        sum_dist = np.inf
        for agent in obs:
            if agent.type == 'officer':
                if sum_dist == np.inf:
                    sum_dist = 0
                sum_dist += get_distance_between(self.limit_board, position, agent.position)
        return sum_dist

    def draw_action(self, obs):
        """
        Choose the position as far away from patrol
        Args:
            obs: information on where are the other agents. List of agents.
        """
        if self.distance_to_officers(obs) == np.inf:
            return 'none'
        directions = possible_directions(self.limit_board, self.position)
        chosen_direction = max(directions, key=lambda direction: self.distance_to_officers(obs, direction))
        return chosen_direction
