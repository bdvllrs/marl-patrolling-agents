import numpy as np
import matplotlib.pyplot as plt
from utils import choice, possible_directions, get_distance_between

__all__ = ["Officer", "Target", "Headquarters"]


class Agent:
    type_id = None
    type = None
    position = None
    color = "red"
    limit_board = None
    view_radius = 10  # manhattan distance
    number_arms = 6  # top, left, right, bottom, top-left, top-right, bottom-left, bottom-right

    histories = []

    def __init__(self, name=None):
        assert (self.type_id is not None and
                self.type is not None), "This agent does not have any type information."
        if name is None:
            name = self.type + " " + str(np.random.randint(0, 1000))
        self.name = name

    def reset(self):
        self.histories.append([])  # New history

    def set_position(self, position):
        self.histories[-1].append(position)  # Keep history of all positions
        self.position = position

    def get_reward(self, reward):
        """
        Get reward for last action
        Args:
            reward:
        """
        pass

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
        possible_positions = possible_directions(self.limit_board, position)
        for new_position in possible_positions:
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

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other agents. List of agents.
        """
        return choice(possible_directions(self.limit_board, self.position))


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

    def distance_to_patrolers(self, obs, position=None):
        position = self.position if position is None else position
        sum_dist = np.inf
        for agent in obs:
            if agent.type == 'patrol':
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
        if self.distance_to_patrolers(obs) == np.inf:
            return self.position
        directions = possible_directions(self.limit_board, self.position)
        position = max(directions, key=lambda pos: self.distance_to_patrolers(obs, pos))
        return position

